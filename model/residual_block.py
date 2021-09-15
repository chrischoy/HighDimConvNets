import torch.nn as nn

from model.common import get_norm, conv

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = 'BN'

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               region_type=ME.RegionType.HYPERCUBE,
               dimension=3):
    super(BasicBlockBase, self).__init__()

    self.conv1 = conv(
        inplanes,
        planes,
        kernel_size=3,
        stride=stride,
        region_type=region_type,
        dimension=dimension)
    self.norm1 = get_norm(
        self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=dimension)
    self.conv2 = conv(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        has_bias=False,
        region_type=region_type,
        dimension=dimension)
    self.norm2 = get_norm(
        self.NORM_TYPE, planes, bn_momentum=bn_momentum, dimension=dimension)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = MEF.relu(out)

    out = self.conv2(out)
    out = self.norm2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = MEF.relu(out)

    return out


class BasicBlockBN(BasicBlockBase):
  NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = 'IN'


class BasicBlockINBN(BasicBlockBase):
  expansion = 1

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               region_type=ME.RegionType.HYPERCUBE,
               dimension=3):
    super(BasicBlockBase, self).__init__()

    self.conv1 = conv(
        inplanes,
        planes,
        kernel_size=3,
        stride=stride,
        region_type=region_type,
        dimension=dimension)
    self.norm1in = get_norm('IN', planes, bn_momentum=bn_momentum, dimension=dimension)
    self.norm1bn = get_norm('BN', planes, bn_momentum=bn_momentum, dimension=dimension)
    self.conv2 = conv(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        has_bias=False,
        region_type=region_type,
        dimension=dimension)
    self.norm2in = get_norm('IN', planes, bn_momentum=bn_momentum, dimension=dimension)
    self.norm2bn = get_norm('BN', planes, bn_momentum=bn_momentum, dimension=dimension)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1in(out)
    out = self.norm1bn(out)
    out = MEF.elu(out)

    out = self.conv2(out)
    out = self.norm2in(out)
    out = self.norm2bn(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = MEF.elu(out)

    return out


class BasicBlockAINBN(BasicBlockBase):
  NORM_TYPE = 'AINBN'


def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              region_type=ME.RegionType.HYPERCUBE,
              dimension=3):
  if norm_type == 'BN':
    return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum,
                        region_type, dimension)
  elif norm_type == 'IN':
    return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum,
                        region_type, dimension)
  elif norm_type == 'INBN':
    return BasicBlockINBN(inplanes, planes, stride, dilation, downsample, bn_momentum,
                          region_type, dimension)
  elif norm_type == 'AINBN':
    return BasicBlockAINBN(inplanes, planes, stride, dilation, downsample, bn_momentum,
                           region_type, dimension)
  else:
    raise ValueError(f'Type {norm_type}, not defined')
