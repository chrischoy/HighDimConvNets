import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from model.common import get_norm, get_nonlinearity, conv, conv_tr, conv_norm_non

from model.residual_block import get_block
from model.resnetsc import SCBlock

from lib.util_2d import compute_e_hat


class PyramidModule(ME.MinkowskiNetwork):
  NONLINEARITY = 'ELU'
  NORM_TYPE = 'BN'
  REGION_TYPE = ME.RegionType.HYPERCUBE

  def __init__(self,
               inc,
               outc,
               inner_inc,
               inner_outc,
               inner_module=None,
               depth=1,
               bn_momentum=0.05,
               dimension=-1):
    ME.MinkowskiNetwork.__init__(self, dimension)
    self.depth = depth

    self.conv = nn.Sequential(
        conv_norm_non(
            inc,
            inner_inc,
            3,
            2,
            dimension,
            region_type=self.REGION_TYPE,
            norm_type=self.NORM_TYPE,
            nonlinearity=self.NONLINEARITY), *[
                get_block(
                    self.NORM_TYPE,
                    inner_inc,
                    inner_inc,
                    bn_momentum=bn_momentum,
                    region_type=self.REGION_TYPE,
                    dimension=dimension) for d in range(depth)
            ])
    self.inner_module = inner_module
    self.convtr = nn.Sequential(
        conv_tr(
            in_channels=inner_outc,
            out_channels=inner_outc,
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            region_type=self.REGION_TYPE,
            dimension=dimension),
        get_norm(
            self.NORM_TYPE, inner_outc, bn_momentum=bn_momentum, dimension=dimension),
        get_nonlinearity(self.NONLINEARITY))

    self.cat_conv = conv_norm_non(
        inner_outc + inc,
        outc,
        1,
        1,
        dimension,
        norm_type=self.NORM_TYPE,
        nonlinearity=self.NONLINEARITY)

  def forward(self, x):
    y = self.conv(x)
    if self.inner_module:
      y = self.inner_module(y)
    y = self.convtr(y)
    y = ME.cat(x, y)
    return self.cat_conv(y)


class PyramidModuleINBN(PyramidModule):
  NORM_TYPE = 'INBN'


class PyramidModuleAINBN(PyramidModule):
  NORM_TYPE = 'AINBN'


class PyramidNet(ME.MinkowskiNetwork):
  NORM_TYPE = 'BN'
  NONLINEARITY = 'ELU'
  PYRAMID_MODULE = PyramidModule
  CHANNELS = [32, 64, 128, 128]
  TR_CHANNELS = [64, 128, 128, 128]
  DEPTHS = [1, 1, 1, 1]
  # None        b1, b2, b3, btr3, btr2
  #               1  2  3 -3 -2 -1
  REGION_TYPE = ME.RegionType.HYPERCUBE

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               conv1_kernel_size=3,
               normalize_feature=False,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    self.conv1_kernel_size = conv1_kernel_size
    self.normalize_feature = normalize_feature

    self.initialize_network(in_channels, out_channels, bn_momentum, D)

  def initialize_network(self, in_channels, out_channels, bn_momentum, dimension):
    NORM_TYPE = self.NORM_TYPE
    NONLINEARITY = self.NONLINEARITY
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    DEPTHS = self.DEPTHS
    REGION_TYPE = self.REGION_TYPE

    self.conv = conv_norm_non(
        in_channels,
        CHANNELS[0],
        kernel_size=self.conv1_kernel_size,
        stride=1,
        dimension=dimension,
        bn_momentum=bn_momentum,
        region_type=REGION_TYPE,
        norm_type=NORM_TYPE,
        nonlinearity=NONLINEARITY)

    pyramid = None
    for d in range(len(DEPTHS) - 1, 0, -1):
      pyramid = self.PYRAMID_MODULE(
          CHANNELS[d - 1],
          TR_CHANNELS[d - 1],
          CHANNELS[d],
          TR_CHANNELS[d],
          pyramid,
          DEPTHS[d],
          dimension=dimension)
    self.pyramid = pyramid
    self.final = nn.Sequential(
        conv_norm_non(
            TR_CHANNELS[0],
            TR_CHANNELS[0],
            kernel_size=3,
            stride=1,
            dimension=dimension),
        conv(TR_CHANNELS[0], out_channels, 1, 1, dimension=dimension))

  def forward(self, x):
    out = self.conv(x)
    out = self.pyramid(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out


class PyramidNet8(PyramidNet):
  CHANNELS = [32, 64, 128, 128, 192, 192, 256, 256]
  TR_CHANNELS = [64, 128, 128, 192, 192, 192, 256, 256]
  DEPTHS = [1, 1, 1, 1, 1, 1, 1, 1]


class PyramidNet8INBN(PyramidNet8):
  NORM_TYPE = 'INBN'
  PYRAMID_MODULE = PyramidModuleINBN


class PyramidNet8AINBN(PyramidNet8):
  NORM_TYPE = 'AINBN'
  PYRAMID_MODULE = PyramidModuleAINBN

class PyramidNet8AINBNNoBlock(PyramidNet8):
  NORM_TYPE = 'AINBN'
  PYRAMID_MODULE = PyramidModuleAINBN
  DEPTHS = [0, 0, 0, 0, 0, 0, 0, 0]

class PyramidNet8NoBlock(PyramidNet8):
  DEPTHS = [0, 0, 0, 0, 0, 0, 0, 0]


class PyramidNet12(PyramidNet):
  CHANNELS = [32, 64, 128, 128, 128, 192, 192, 192, 256, 256, 256, 512]
  TR_CHANNELS = [64, 64, 128, 128, 128, 192, 192, 192, 256, 256, 256, 512]
  DEPTHS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


class PyramidNet12INBN(PyramidNet12):
  NORM_TYPE = 'INBN'
  PYRAMID_MODULE = PyramidModuleINBN


class PyramidNet12AINBN(PyramidNet12):
  NORM_TYPE = 'AINBN'
  PYRAMID_MODULE = PyramidModuleAINBN

class PyramidNet12AINBNNoBlock(PyramidNet12AINBN):
  DEPTHS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# --sample_minimum_coords False
class PyramidIterationNet(PyramidNet):
  PYRAMID_NET = PyramidNet8INBN

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.05,
               conv1_kernel_size=3,
               normalize_feature=False,
               D=4):
    ME.MinkowskiNetwork.__init__(self, D)
    self.conv1_kernel_size = conv1_kernel_size

    self.pyramid1 = self.PYRAMID_NET(
        in_channels,
        out_channels,
        bn_momentum,
        conv1_kernel_size,
        normalize_feature=False,
        D=D)
    self.pyramid2 = self.PYRAMID_NET(
        in_channels + 2,
        out_channels,
        bn_momentum,
        conv1_kernel_size,
        normalize_feature=False,
        D=D)

  def forward(self, x, data):
    xyz, len_batch = data['xyz'], data['len_batch']

    out = self.pyramid1(x)
    logits = out.F.squeeze()
    e_hats, residuals = compute_e_hat(xyz, logits, len_batch)

    new_feat = torch.cat([
        x.feats,
        residuals.detach().float().to(logits.device).unsqueeze(1),
        F.relu(torch.tanh(logits)).detach().unsqueeze(1)
    ],
                         dim=1)
    x_iter = ME.SparseTensor(
        feats=new_feat, coords_key=x.coords_key,
        coords_manager=x.coords_man).to(logits.device)

    out_iter = self.pyramid2(x_iter)
    logits_iter = out_iter.F.squeeze()
    e_hats_iter, residuals_iter = compute_e_hat(xyz, logits_iter, len_batch)
    res_logits, res_e_hats = [logits, logits_iter], [e_hats, e_hats_iter]
    return res_logits, res_e_hats


class PyramidIterationNetNoBlock(PyramidIterationNet):
  PYRAMID_NET = PyramidNet8NoBlock


class PyramidNetSC(PyramidNet):
  CHANNELS = [32, 64, 128, 128, 192, 192, 256, 256]
  TR_CHANNELS = [128, 128, 128, 192, 192, 192, 256, 256]
  DEPTHS = [1, 1, 1, 1, 1, 1, 1, 1]
  NORM_TYPE = 'INBN'
  PYRAMID_MODULE = PyramidModuleINBN

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.05,
               conv1_kernel_size=3,
               depth=6,
               clusters=500,
               D=4):
    ME.MinkowskiNetwork.__init__(self, D)
    self.conv1_kernel_size = conv1_kernel_size

    self.initialize_network(in_channels, out_channels, bn_momentum, D)
    self.initialize_scblocks(
        in_channels + 2, out_channels, depth=depth, clusters=clusters, D=D)

  def initialize_scblocks(self, in_channels, out_channels, depth, clusters, D):
    self.scblock = SCBlock(
        in_channels, out_channels, depth=depth, clusters=clusters, D=D)

  def forward(self, x, data):
    xyz, len_batch = data['xyz'], data['len_batch']

    out = self.conv(x)
    out = self.pyramid(out)
    out = self.final(out)

    logits = out.F.squeeze()
    e_hats, residuals = compute_e_hat(xyz, logits, len_batch)

    new_feat = torch.cat([
        x.feats,
        residuals.detach().float().to(logits.device).unsqueeze(1),
        F.relu(torch.tanh(logits)).detach().unsqueeze(1)
    ],
                         dim=1)
    x_iter = ME.SparseTensor(
        feats=new_feat, coords_key=x.coords_key,
        coords_manager=x.coords_man).to(logits.device)

    logits_iter, e_hats_iter, residuals_iter = self.scblock(x_iter, data)

    res_logits, res_e_hats = [logits, logits_iter], [e_hats, e_hats_iter]
    return res_logits, res_e_hats


class PyramidNetSCNoBlock(PyramidNetSC):
  DEPTHS = [0, 0, 0, 0, 0, 0, 0, 0]
