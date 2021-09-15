import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.model.oanet import OAFilter, diff_pool, diff_unpool
from lib.util_2d import compute_e_hat
from model.common import conv, conv_tr, get_norm


class BasicBlock(ME.MinkowskiNetwork):

  def __init__(self, in_channels, out_channels=None, D=4, transpose=False, stride=1):
    ME.MinkowskiNetwork.__init__(self, D)
    if not out_channels:
      out_channels = in_channels
    self.shot_cut = None
    if out_channels != in_channels:
      self.shot_cut = conv(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=1,
          dimension=D)
    if transpose:
      self.conv = nn.Sequential(
          get_norm('IN', in_channels, bn_momentum=0.1, dimension=D),
          get_norm('BN', in_channels, bn_momentum=0.1, dimension=D),
          conv_tr(in_channels, out_channels, kernel_size=3, stride=stride, dimension=D),
          get_norm('IN', out_channels, bn_momentum=0.1, dimension=D),
          get_norm('BN', out_channels, bn_momentum=0.1, dimension=D),
          ME.MinkowskiReLU(),
          conv_tr(
              out_channels, out_channels, kernel_size=3, stride=stride, dimension=D))
    else:
      self.conv = nn.Sequential(
          get_norm('IN', in_channels, bn_momentum=0.1, dimension=D),
          get_norm('BN', in_channels, bn_momentum=0.1, dimension=D),
          conv(in_channels, out_channels, kernel_size=3, stride=stride, dimension=D),
          get_norm('IN', out_channels, bn_momentum=0.1, dimension=D),
          get_norm('BN', out_channels, bn_momentum=0.1, dimension=D),
          ME.MinkowskiReLU(),
          conv(out_channels, out_channels, kernel_size=3, stride=stride, dimension=D))

  def forward(self, x):
    out = self.conv(x)
    if self.shot_cut:
      out = out + self.shot_cut(x)
    else:
      out = out + x
    return out


class DiffPool(diff_pool):

  def forward(self, x, len_batch):
    num_points = len_batch[0]
    batch_size = len(len_batch)
    assert len_batch.count(
        num_points) == batch_size, f'batch contains different numbers of coordinates'
    # x: (n, c), n = m*b
    input = x.reshape(batch_size, num_points, -1)  # input: (b,m,c)
    input = input.transpose(1, 2).unsqueeze(-1)  # input: (b,c,m,1)
    embed = self.conv(input)  # embed: (b,k,m,1)
    S = torch.softmax(embed, dim=2).squeeze(3)  # (b,k,m)
    out = torch.matmul(input.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
    return out


class DiffUnpool(diff_unpool):

  def forward(self, x_up, x_down, len_batch):
    num_points = len_batch[0]
    batch_size = len(len_batch)
    assert len_batch.count(
        num_points) == batch_size, f'batch contains different numbers of coordinates'

    input = x_up.reshape(batch_size, num_points, -1)  # input: (b,m,c)
    input = input.transpose(1, 2).unsqueeze(-1)  # input: (b,c,m,1)
    embed = self.conv(input)  # embed: (b,k,m,1)
    S = torch.softmax(embed, dim=1).squeeze(3)  # (b,k,m)
    out = torch.matmul(x_down.squeeze(3), S)  # (b,c,k) * (b,k,m) => (b,c,m)
    num_channel = out.shape[1]
    out = out.transpose(1, 2).reshape(-1, num_channel)
    return out


class SCBlock(ME.MinkowskiNetwork):
  """Spatial Correlation Block"""
  NET_CHANNEL = 128

  def __init__(self, in_channels, out_channels, depth, clusters, D=4):
    ME.MinkowskiNetwork.__init__(self, D)
    self.depth = depth
    self.clusters = clusters
    net_channels = self.NET_CHANNEL

    self.conv1 = conv(in_channels, net_channels, kernel_size=1, dimension=D)

    self.l1_1 = []
    for _ in range(depth // 2):
      self.l1_1.append(BasicBlock(in_channels=net_channels, D=D))

    self.down1 = DiffPool(net_channels, clusters)

    self.l2 = []
    for _ in range(depth // 2):
      self.l2.append(OAFilter(net_channels, clusters))

    self.up1 = DiffUnpool(net_channels, clusters)

    self.l1_2 = []
    self.l1_2.append(BasicBlock(2 * net_channels, net_channels, D=D, transpose=True))
    for _ in range(depth // 2 - 1):
      self.l1_2.append(BasicBlock(net_channels, net_channels, D=D, transpose=True))

    self.l1_1 = nn.Sequential(*self.l1_1)
    self.l1_2 = nn.Sequential(*self.l1_2)
    self.l2 = nn.Sequential(*self.l2)
    self.output = conv(net_channels, out_channels, kernel_size=1, dimension=D)

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiConvolution):
        ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

      if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

  def forward(self, x, data):
    xyz, len_batch = data['xyz'], data['len_batch']
    x1_1 = self.conv1(x)
    x1_1 = self.l1_1(x1_1)
    x1_1_F = x1_1.F

    x_down = self.down1(x1_1_F, len_batch)
    x2 = self.l2(x_down)
    x_up = self.up1(x1_1_F, x2, len_batch)
    x_up = ME.SparseTensor(
        x_up,
        coords_key=x1_1.coords_key,
        coords_manager=x1_1.coords_man,
    )
    x = ME.cat(x1_1, x_up)
    out = self.l1_2(x)
    out = self.output(out)

    logits = out.F.squeeze()
    e_hats, residuals = compute_e_hat(xyz, logits, len_batch)

    return logits, e_hats, residuals


class ResNetSC(ME.MinkowskiNetwork):
  BLOCK = SCBlock

  def __init__(self, in_channels, out_channels=None, clusters=None, D=4):
    ME.MinkowskiNetwork.__init__(self, D)

    self.iter_num = 1
    self.depth = 6
    self.clusters = clusters
    self.weight_init = self.BLOCK(
        in_channels,
        out_channels,
        self.depth,
        self.clusters,
        D,
    )
    self.weight_iter = [
        self.BLOCK(
            in_channels + 2,
            out_channels,
            self.depth,
            self.clusters,
            D,
        ) for _ in range(self.iter_num)
    ]
    self.weight_iter = nn.Sequential(*self.weight_iter)

  def forward(self, x, data):
    res_logits, res_e_hats = [], []

    logits, e_hat, residual = self.weight_init(x, data)
    res_logits.append(logits)
    res_e_hats.append(e_hat)

    for i in range(self.iter_num):
      new_feat = torch.cat([
          x.feats,
          residual.detach().float().to(logits.device).unsqueeze(1),
          F.relu(torch.tanh(logits)).detach().unsqueeze(1)
      ],
                           dim=1)
      new_tensor = ME.SparseTensor(
          new_feat, coords_key=x.coords_key,
          coords_manager=x.coords_man).to(logits.device)
      logits, e_hat, residual = self.weight_iter[i](new_tensor, data)
      res_logits.append(logits)
      res_e_hats.append(e_hat)

    return res_logits, res_e_hats
