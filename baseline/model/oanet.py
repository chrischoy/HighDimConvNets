import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.util_2d import batch_episym, torch_skew_symmetric, weighted_8points


class OALoss(object):

  def __init__(self, config):
    self.loss_essential = config.oa_loss_essential
    self.loss_classif = config.oa_loss_classif
    self.use_fundamental = config.oa_use_fundamental
    self.obj_geod_th = config.oa_obj_geod_th
    self.geo_loss_margin = config.oa_geo_loss_margin
    self.loss_essential_init_iter = config.oa_loss_essential_init_iter

  def run(self, global_step, data, logits, e_hat):
    e_gt_unnorm, labels, pts_virt = data['E'], data['labels'], data['virtPts']
    e_gt_unnorm = torch.reshape(e_gt_unnorm, (-1, 9))

    # Essential/Fundamental matrix loss
    pts1_virts, pts2_virts = pts_virt[:, :, :2], pts_virt[:, :, 2:]
    geod = batch_episym(pts1_virts, pts2_virts, e_hat)
    essential_loss = torch.min(geod, self.geo_loss_margin * geod.new_ones(geod.shape))
    essential_loss = essential_loss.mean()

    # Classification loss
    is_pos = labels.to(torch.bool)
    is_neg = ~is_pos
    is_pos = is_pos.to(logits.dtype)
    is_neg = is_neg.to(logits.dtype)
    c = is_pos - is_neg
    classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())
    # balance
    num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
    num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
    classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
    classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)
    classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos +
                              classif_loss_n * 0.5 / num_neg)

    loss = 0
    # Check global_step and add essential loss
    if self.loss_essential > 0 and global_step >= self.loss_essential_init_iter:
      loss += self.loss_essential * essential_loss
    if self.loss_classif > 0:
      loss += self.loss_classif * classif_loss

    return loss


class PointCN(nn.Module):

  def __init__(self, channels, out_channels=None):
    nn.Module.__init__(self)
    if not out_channels:
      out_channels = channels
    self.shot_cut = None
    if out_channels != channels:
      self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
    self.conv = nn.Sequential(
        nn.InstanceNorm2d(channels, eps=1e-3), nn.BatchNorm2d(channels), nn.ReLU(),
        nn.Conv2d(channels, out_channels, kernel_size=1),
        nn.InstanceNorm2d(out_channels, eps=1e-3), nn.BatchNorm2d(out_channels),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1))

  def forward(self, x):
    out = self.conv(x)
    if self.shot_cut:
      out = out + self.shot_cut(x)
    else:
      out = out + x
    return out


class trans(nn.Module):

  def __init__(self, dim1, dim2):
    nn.Module.__init__(self)
    self.dim1 = dim1
    self.dim2 = dim2

  def forward(self, x):
    return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):

  def __init__(self, channels, points, out_channels=None):
    nn.Module.__init__(self)
    if not out_channels:
      out_channels = channels
    self.shot_cut = None
    if out_channels != channels:
      self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
    self.conv1 = nn.Sequential(
        nn.InstanceNorm2d(channels, eps=1e-3),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.Conv2d(channels, out_channels, kernel_size=1),  #b*c*n*1
        trans(1, 2))
    # Spatial Correlation Layer
    self.conv2 = nn.Sequential(
        nn.BatchNorm2d(points), nn.ReLU(), nn.Conv2d(points, points, kernel_size=1))
    self.conv3 = nn.Sequential(
        trans(1, 2), nn.InstanceNorm2d(out_channels, eps=1e-3),
        nn.BatchNorm2d(out_channels), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1))

  def forward(self, x):
    out = self.conv1(x)
    out = out + self.conv2(out)
    out = self.conv3(out)
    if self.shot_cut:
      out = out + self.shot_cut(x)
    else:
      out = out + x
    return out


# you can use this bottleneck block to prevent from overfiting when your dataset is small
class OAFilterBottleneck(nn.Module):

  def __init__(self, channels, points1, points2, out_channels=None):
    nn.Module.__init__(self)
    if not out_channels:
      out_channels = channels
    self.shot_cut = None
    if out_channels != channels:
      self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
    self.conv1 = nn.Sequential(
        nn.InstanceNorm2d(channels, eps=1e-3),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.Conv2d(channels, out_channels, kernel_size=1),  #b*c*n*1
        trans(1, 2))
    self.conv2 = nn.Sequential(
        nn.BatchNorm2d(points1), nn.ReLU(), nn.Conv2d(points1, points2, kernel_size=1),
        nn.BatchNorm2d(points2), nn.ReLU(), nn.Conv2d(points2, points1, kernel_size=1))
    self.conv3 = nn.Sequential(
        trans(1, 2), nn.InstanceNorm2d(out_channels, eps=1e-3),
        nn.BatchNorm2d(out_channels), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1))

  def forward(self, x):
    out = self.conv1(x)
    out = out + self.conv2(out)
    out = self.conv3(out)
    if self.shot_cut:
      out = out + self.shot_cut(x)
    else:
      out = out + x
    return out


class diff_pool(nn.Module):

  def __init__(self, in_channel, output_points):
    nn.Module.__init__(self)
    self.output_points = output_points
    self.conv = nn.Sequential(
        nn.InstanceNorm2d(in_channel, eps=1e-3), nn.BatchNorm2d(in_channel), nn.ReLU(),
        nn.Conv2d(in_channel, output_points, kernel_size=1))

  def forward(self, x):
    embed = self.conv(x)  # b*k*n*1
    S = torch.softmax(embed, dim=2).squeeze(3)
    out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
    return out


class diff_unpool(nn.Module):

  def __init__(self, in_channel, output_points):
    nn.Module.__init__(self)
    self.output_points = output_points
    self.conv = nn.Sequential(
        nn.InstanceNorm2d(in_channel, eps=1e-3), nn.BatchNorm2d(in_channel), nn.ReLU(),
        nn.Conv2d(in_channel, output_points, kernel_size=1))

  def forward(self, x_up, x_down):
    #x_up: b*c*n*1
    #x_down: b*c*k*1
    embed = self.conv(x_up)  # b*k*n*1
    S = torch.softmax(embed, dim=1).squeeze(3)  # b*k*n
    out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
    return out  # b*c*n*1


class OANBlock(nn.Module):

  def __init__(self, net_channels, input_channel, depth, clusters):
    nn.Module.__init__(self)
    channels = net_channels
    self.layer_num = depth
    print('channels:' + str(channels) + ', layer_num:' + str(self.layer_num))
    self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

    l2_nums = clusters

    self.l1_1 = []
    for _ in range(self.layer_num // 2):
      self.l1_1.append(PointCN(channels))

    self.down1 = diff_pool(channels, l2_nums)

    self.l2 = []
    for _ in range(self.layer_num // 2):
      self.l2.append(OAFilter(channels, l2_nums))

    self.up1 = diff_unpool(channels, l2_nums)

    self.l1_2 = []
    self.l1_2.append(PointCN(2 * channels, channels))
    for _ in range(self.layer_num // 2 - 1):
      self.l1_2.append(PointCN(channels))

    self.l1_1 = nn.Sequential(*self.l1_1)
    self.l1_2 = nn.Sequential(*self.l1_2)
    self.l2 = nn.Sequential(*self.l2)

    self.output = nn.Conv2d(channels, 1, kernel_size=1)

  def forward(self, data, xs):
    #data: b*c*n*1
    batch_size, num_pts = data.shape[0], data.shape[2]
    x1_1 = self.conv1(data)
    x1_1 = self.l1_1(x1_1)
    x_down = self.down1(x1_1)
    x2 = self.l2(x_down)
    x_up = self.up1(x1_1, x2)
    out = self.l1_2(torch.cat([x1_1, x_up], dim=1))

    logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
    e_hat = weighted_8points(xs.squeeze(1).permute(0, 2, 1), logits)

    x1, x2 = xs[:, 0, :, :2], xs[:, 0, :, 2:4]
    e_hat_norm = e_hat
    residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

    return logits, e_hat, residual


class OANet(nn.Module):

  def __init__(self, config):
    nn.Module.__init__(self)
    self.iter_num = config.oa_iter_num
    depth_each_stage = config.oa_net_depth // (config.oa_iter_num + 1)
    self.side_channel = (config.oa_use_ratio == 2) + (config.oa_use_mutual == 2)
    self.weights_init = OANBlock(config.oa_net_channels, 4 + self.side_channel,
                                 depth_each_stage, config.oa_clusters)
    self.weights_iter = [
        OANBlock(config.oa_net_channels, 6 + self.side_channel, depth_each_stage,
                 config.oa_clusters) for _ in range(config.oa_iter_num)
    ]
    self.weights_iter = nn.Sequential(*self.weights_iter)
    self.config = config

  def forward(self, x):
    assert x.dim() == 4 and x.shape[1] == 1

    input = x.transpose(1, 3)

    res_logits, res_e_hat = [], []
    logits, e_hat, residual = self.weights_init(input, x)
    res_logits.append(logits), res_e_hat.append(e_hat)

    # For iterative network
    for i in range(self.iter_num):
      logits, e_hat, residual = self.weights_iter[i](torch.cat([
          input,
          residual.detach(),
          F.relu(torch.tanh(logits)).reshape(residual.shape).detach()
      ],
                                                               dim=1), x)
      res_logits.append(logits), res_e_hat.append(e_hat)
    return res_logits, res_e_hat