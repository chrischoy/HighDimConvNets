import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.util_2d import weighted_8points


class ClassificationLoss(nn.Module):

  def forward(self, logits, labels):
    is_pos = labels.to(torch.bool)
    is_neg = (~is_pos)
    is_pos = is_pos.to(torch.float)
    is_neg = is_neg.to(torch.float)
    c = is_pos - is_neg

    loss = -F.logsigmoid(c * logits)
    num_pos = F.relu(torch.sum(is_pos, dim=1) - 1) + 1
    num_neg = F.relu(torch.sum(is_neg, dim=1) - 1) + 1

    loss_pos = torch.sum(loss * is_pos, dim=1)
    loss_neg = torch.sum(loss * is_neg, dim=1)

    balanced_loss = torch.mean(loss_pos * 0.5 / num_pos + loss_neg * 0.5 / num_neg)
    return balanced_loss


class RegressionLoss(nn.Module):

  def forward(self, logits, coords, e_gt):
    e = weighted_8points(coords, logits)
    e_gt = torch.reshape(e_gt, (logits.shape[0], 9))
    e_gt = e_gt / torch.norm(e_gt, dim=1, keepdim=True)

    loss = torch.mean(
        torch.min(torch.sum((e - e_gt)**2, dim=1), torch.sum((e + e_gt)**2, dim=1)))
    return loss


class LFGCLoss(nn.Module):

  def __init__(self, alpha, beta, regression_iter):
    super(LFGCLoss, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.regression_iter = regression_iter

  def forward(self, logits, coords, labels, e_gt, iteration):
    ClsLoss = ClassificationLoss()
    RegLoss = RegressionLoss()

    cls_loss = ClsLoss(logits, labels)

    if iteration > self.regression_iter:
      reg_loss = RegLoss(logits, coords, e_gt)
      loss = cls_loss * self.alpha + reg_loss * self.beta
    else:
      loss = cls_loss * self.alpha

    return loss


class ContextNorm(nn.Module):

  def __init__(self, eps):
    super(ContextNorm, self).__init__()
    self.eps = eps

  def forward(self, x):
    variance, mean = torch.var_mean(x, dim=2, keepdim=True)
    std = torch.sqrt(variance)
    return (x - mean) / (std + self.eps)


class ResNetBlock(nn.Module):

  def __init__(self, in_channel, out_channel, kernel_size, stride):
    super(ResNetBlock, self).__init__()
    self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, stride)
    self.cn1 = ContextNorm(eps=1e-3)
    self.bn1 = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.99)
    self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size, stride)
    self.cn2 = ContextNorm(eps=1e-3)
    self.bn2 = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.99)

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.cn1(out)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.conv2(out)
    out = self.cn2(out)
    out = self.bn2(out)
    out = F.relu(out)

    return out + residual


class LFGCNet(nn.Module):
  """LFGCNet

  This model need normalized correspondences(4D) as input 
  Input shape should be (batch_size, 4, num_point)

  """

  def __init__(self, in_channel=4, out_channel=128, depth=12, config=None):
    super(LFGCNet, self).__init__()
    self.input = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1)

    blocks = [
        ResNetBlock(out_channel, out_channel, kernel_size=1, stride=1)
        for _ in range(depth)
    ]
    self.blocks = nn.Sequential(*blocks)

    self.output = nn.Conv1d(out_channel, 1, kernel_size=1, stride=1)

    self.config = config

  def forward(self, x):
    out = self.input(x)
    out = self.blocks(out)
    out = self.output(out)

    return out