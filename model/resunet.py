import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.common import get_norm, conv, conv_tr

from model.residual_block import get_block


class ResUNet(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128]
  TR_CHANNELS = [None, 32, 64, 64]
  # None        b1, b2, b3, btr3, btr2
  #               1  2  3 -3 -2 -1
  DEPTHS = [None, 1, 1, 1, 1, 1, None]
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
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    DEPTHS = self.DEPTHS
    REGION_TYPE = self.REGION_TYPE
    self.normalize_feature = normalize_feature
    self.conv1 = conv(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.block1 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[1],
            CHANNELS[1],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[1])
    ])

    self.conv2 = conv(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[2],
            CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[2])
    ])

    self.conv3 = conv(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[3],
            CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[3])
    ])

    self.conv3_tr = conv_tr(
        in_channels=CHANNELS[3],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[3],
            TR_CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-3])
    ])

    self.conv2_tr = conv_tr(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[2],
            TR_CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-2])
    ])

    self.conv1_tr = conv_tr(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.final = conv(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = MEF.relu(out_s1)
    out_s1 = self.block1(out_s1)

    out_s2 = self.conv2(out_s1)
    out_s2 = self.norm2(out_s2)
    out_s2 = MEF.relu(out_s2)
    out_s2 = self.block2(out_s2)

    out_s4 = self.conv3(out_s2)
    out_s4 = self.norm3(out_s4)
    out_s4 = MEF.relu(out_s4)
    out_s4 = self.block3(out_s4)

    out_s2t = self.conv3_tr(out_s4)
    out_s2t = self.norm3_tr(out_s2t)
    out_s2t = MEF.relu(out_s2t)
    out_s2t = self.block3_tr(out_s2t)

    out = ME.cat(out_s2t, out_s2)

    out_s1t = self.conv2_tr(out)
    out_s1t = self.norm2_tr(out_s1t)
    out_s1t = MEF.relu(out_s1t)
    out_s1t = self.block2_tr(out_s1t)

    out = ME.cat(out_s1t, out_s1)

    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out


class ResUNetBN(ResUNet):
  NORM_TYPE = 'BN'


class ResUNetBNF(ResUNet):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64]
  TR_CHANNELS = [None, 16, 32, 64]


class ResUNetBNFx2(ResUNet):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64]
  TR_CHANNELS = [None, 16, 32, 64]
  # None        b1, b2, b3, btr3, btr2
  #               1  2  3 -3 -2 -1
  DEPTHS = [None, 1, 2, 2, 2, 1, None]


class ResUNetBNFx3(ResUNet):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64]
  TR_CHANNELS = [None, 16, 32, 64]
  # None        b1, b2, b3, btr3, btr2
  #               1  2  3 -3 -2 -1
  DEPTHS = [None, 1, 3, 3, 3, 1, None]


class ResUNet2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]
  # None        b1, b2, b3, b4, btr4, btr3, btr2
  #               1  2  3  4,-4,-3,-2,-1
  DEPTHS = [None, 1, 1, 1, 1, 1, 1, 1, None]
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
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    DEPTHS = self.DEPTHS
    REGION_TYPE = self.REGION_TYPE
    self.normalize_feature = normalize_feature
    self.conv1 = conv(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.block1 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[1],
            CHANNELS[1],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[1])
    ])

    self.conv2 = conv(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[2],
            CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[2])
    ])

    self.conv3 = conv(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[3],
            CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[3])
    ])

    self.conv4 = conv(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[4],
            CHANNELS[4],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[4])
    ])

    self.conv4_tr = conv_tr(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm4_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[4],
            TR_CHANNELS[4],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-4])
    ])

    self.conv3_tr = conv_tr(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[3],
            TR_CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-3])
    ])

    self.conv2_tr = conv_tr(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2_tr = get_norm(
        NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[2],
            TR_CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-2])
    ])

    self.conv1_tr = conv_tr(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.final = conv(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)
    self.weight_initialization()

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiConvolution):
        ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

      if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

  def forward(self, x):  # Receptive field size
    out_s1 = self.conv1(x)  # 7
    out_s1 = self.norm1(out_s1)
    out_s1 = MEF.relu(out_s1)
    out_s1 = self.block1(out_s1)

    out_s2 = self.conv2(out_s1)  # 7 + 2 * 2 = 11
    out_s2 = self.norm2(out_s2)
    out_s2 = MEF.relu(out_s2)
    out_s2 = self.block2(out_s2)  # 11 + 2 * (2 + 2) = 19

    out_s4 = self.conv3(out_s2)  # 19 + 4 * 2 = 27
    out_s4 = self.norm3(out_s4)
    out_s4 = MEF.relu(out_s4)
    out_s4 = self.block3(out_s4)  # 27 + 4 * (2 + 2) = 43

    out_s8 = self.conv4(out_s4)  # 43 + 8 * 2 = 59
    out_s8 = self.norm4(out_s8)
    out_s8 = MEF.relu(out_s8)
    out_s8 = self.block4(out_s8)  # 59 + 8 * (2 + 2) = 91

    out = self.conv4_tr(out_s8)  # 91 + 4 * 2 = 99
    out = self.norm4_tr(out)
    out = MEF.relu(out)
    out = self.block4_tr(out)  # 99 + 4 * (2 + 2) = 115

    out = ME.cat(out, out_s4)

    out = self.conv3_tr(out)  # 115 + 2 * 2 = 119
    out = self.norm3_tr(out)
    out = MEF.relu(out)
    out = self.block3_tr(out)  # 119 + 2 * (2 + 2) = 127

    out = ME.cat(out, out_s2)

    out = self.conv2_tr(out)  # 127 + 2 = 129
    out = self.norm2_tr(out)
    out = MEF.relu(out)
    out = self.block2_tr(out)  # 129 + 1 * (2 + 2) = 133

    out = ME.cat(out, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out


class ResUNetBN2(ResUNet2):
  NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetBN2F(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128]
  TR_CHANNELS = [None, 16, 32, 64, 128]


class ResUNetBN2FC(ResUNetBN2F):
  REGION_TYPE = ME.RegionType.HYPERCROSS


class ResUNetBN2Fx2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128]
  TR_CHANNELS = [None, 16, 32, 64, 128]
  # None        b1, b2, b3, b4, btr4, btr3, btr2
  #               1  2  3  4,-4,-3,-2,-1
  DEPTHS = [None, 1, 2, 2, 2, 2, 2, 1, None]


class ResUNetBN2Fx3(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128]
  TR_CHANNELS = [None, 16, 32, 64, 128]
  # None        b1, b2, b3, b4, btr4, btr3, btr2
  #               1  2  3  4,-4,-3,-2,-1
  DEPTHS = [None, 1, 3, 3, 3, 3, 3, 1, None]


class ResUNetBN2G(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 128]
  TR_CHANNELS = [None, 128, 128, 128, 128]


class ResUNetBN2GC(ResUNetBN2G):
  REGION_TYPE = ME.RegionType.HYPERCROSS


class ResUNetIN2GC(ResUNetBN2GC):
  NORM_TYPE = 'IN'


class ResUNetBN2Gx2(ResUNetBN2G):
  DEPTHS = [None, 1, 2, 2, 2, 2, 2, 1, None]


class ResUNetBN2Gx2C(ResUNetBN2Gx2):
  REGION_TYPE = ME.RegionType.HYPERCROSS


class ResUNetINBN2(ResUNet2):
  BLOCK_NORM_TYPE = 'INBN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]
  # None        b1, b2, b3, b4, btr4, btr3, btr2
  #               1  2  3  4,-4,-3,-2,-1
  DEPTHS = [None, 1, 1, 1, 1, 1, 1, 1, None]
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
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    DEPTHS = self.DEPTHS
    REGION_TYPE = self.REGION_TYPE
    self.normalize_feature = normalize_feature
    self.conv1 = conv(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm1in = get_norm('IN', CHANNELS[1], bn_momentum=bn_momentum, dimension=D)
    self.norm1bn = get_norm('BN', CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.block1 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[1],
            CHANNELS[1],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[1])
    ])

    self.conv2 = conv(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2in = get_norm('IN', CHANNELS[2], bn_momentum=bn_momentum, dimension=D)
    self.norm2bn = get_norm('BN', CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[2],
            CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[2])
    ])

    self.conv3 = conv(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3in = get_norm('IN', CHANNELS[3], bn_momentum=bn_momentum, dimension=D)
    self.norm3bn = get_norm('BN', CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[3],
            CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[3])
    ])

    self.conv4 = conv(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm4in = get_norm('IN', CHANNELS[4], bn_momentum=bn_momentum, dimension=D)
    self.norm4bn = get_norm('BN', CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4 = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[4],
            CHANNELS[4],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[4])
    ])

    self.conv4_tr = conv_tr(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm4in_tr = get_norm(
        'IN', TR_CHANNELS[4], bn_momentum=bn_momentum, dimension=D)
    self.norm4bn_tr = get_norm(
        'BN', TR_CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

    self.block4_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[4],
            TR_CHANNELS[4],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-4])
    ])

    self.conv3_tr = conv_tr(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm3in_tr = get_norm(
        'IN', TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)
    self.norm3bn_tr = get_norm(
        'BN', TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

    self.block3_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[3],
            TR_CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-3])
    ])

    self.conv2_tr = conv_tr(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)
    self.norm2in_tr = get_norm(
        'IN', TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)
    self.norm2bn_tr = get_norm(
        'BN', TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

    self.block2_tr = nn.Sequential(*[
        get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[2],
            TR_CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D) for d in range(DEPTHS[-2])
    ])

    self.conv1_tr = conv_tr(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=False,
        region_type=REGION_TYPE,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

    self.final = conv(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)
    self.weight_initialization()

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiConvolution):
        ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

      if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1in(out_s1)
    out_s1 = self.norm1bn(out_s1)
    out_s1 = MEF.relu(out_s1)
    out_s1 = self.block1(out_s1)

    out_s2 = self.conv2(out_s1)
    out_s2 = self.norm2in(out_s2)
    out_s2 = self.norm2bn(out_s2)
    out_s2 = MEF.relu(out_s2)
    out_s2 = self.block2(out_s2)

    out_s4 = self.conv3(out_s2)
    out_s4 = self.norm3in(out_s4)
    out_s4 = self.norm3bn(out_s4)
    out_s4 = MEF.relu(out_s4)
    out_s4 = self.block3(out_s4)

    out_s8 = self.conv4(out_s4)
    out_s8 = self.norm4in(out_s8)
    out_s8 = self.norm4bn(out_s8)
    out_s8 = MEF.relu(out_s8)
    out_s8 = self.block4(out_s8)

    out = self.conv4_tr(out_s8)
    out = self.norm4in_tr(out)
    out = self.norm4bn_tr(out)
    out = MEF.relu(out)
    out = self.block4_tr(out)

    out = ME.cat(out, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3in_tr(out)
    out = self.norm3bn_tr(out)
    out = MEF.relu(out)
    out = self.block3_tr(out)

    out = ME.cat(out, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2in_tr(out)
    out = self.norm2bn_tr(out)
    out = MEF.relu(out)
    out = self.block2_tr(out)

    out = ME.cat(out, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out


class ResUNetINBN2GP(ResUNetINBN2):
  USE_GPOOL_LINEAR = False
  USE_GPOOL2 = False
  USE_GPOOL3 = False

  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               conv1_kernel_size=3,
               normalize_feature=False,
               D=3):
    ResUNetINBN2.__init__(
        self,
        in_channels,
        out_channels,
        bn_momentum,
        conv1_kernel_size,
        normalize_feature,
        D=D)

    self.gpool1_linear = conv(
        in_channels=self.CHANNELS[1],
        out_channels=self.CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)
    self.gpool1 = ME.MinkowskiGlobalPooling(dimension=D)
    self.bsum1 = ME.MinkowskiBroadcastAddition(dimension=D)

    if self.USE_GPOOL2:
      self.gpool2_linear = conv(
          in_channels=self.CHANNELS[2],
          out_channels=self.CHANNELS[2],
          kernel_size=1,
          stride=1,
          dilation=1,
          has_bias=True,
          dimension=D)

      self.gpool2 = ME.MinkowskiGlobalPooling(dimension=D)
      self.bsum2 = ME.MinkowskiBroadcastAddition(dimension=D)

    if self.USE_GPOOL3:
      self.gpool3_linear = conv(
          in_channels=self.CHANNELS[3],
          out_channels=self.CHANNELS[3],
          kernel_size=1,
          stride=1,
          dilation=1,
          has_bias=True,
          dimension=D)

      self.gpool3 = ME.MinkowskiGlobalPooling(dimension=D)
      self.bsum3 = ME.MinkowskiBroadcastAddition(dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1in(out_s1)
    out_s1 = self.norm1bn(out_s1)
    out_s1 = MEF.relu(out_s1)
    out_s1 = self.block1(out_s1)

    gs1 = out_s1
    if self.USE_GPOOL_LINEAR:
      gs1 = self.gpool1_linear(gs1)

    gs1 = self.gpool1(gs1)
    out_s1 = self.bsum1(out_s1, gs1)

    out_s2 = self.conv2(out_s1)
    out_s2 = self.norm2in(out_s2)
    out_s2 = self.norm2bn(out_s2)
    out_s2 = MEF.relu(out_s2)
    out_s2 = self.block2(out_s2)

    if self.USE_GPOOL2:
      gs2 = out_s2
      if self.USE_GPOOL_LINEAR:
        gs2 = self.gpool2_linear(gs2)

      gs2 = self.gpool2(gs2)
      out_s2 = self.bsum2(out_s2, gs2)

    out_s4 = self.conv3(out_s2)
    out_s4 = self.norm3in(out_s4)
    out_s4 = self.norm3bn(out_s4)
    out_s4 = MEF.relu(out_s4)
    out_s4 = self.block3(out_s4)

    if self.USE_GPOOL3:
      gs4 = out_s4
      if self.USE_GPOOL_LINEAR:
        gs4 = self.gpool3_linear(gs4)

      gs4 = self.gpool3(out_s4)
      out_s4 = self.bsum3(out_s4, gs4)

    out_s8 = self.conv4(out_s4)
    out_s8 = self.norm4in(out_s8)
    out_s8 = self.norm4bn(out_s8)
    out_s8 = MEF.relu(out_s8)
    out_s8 = self.block4(out_s8)

    # gs8 = self.gpool3(out_s8)
    # out_s8 = self.bsum3(out_s8, gs8)

    out = self.conv4_tr(out_s8)
    out = self.norm4in_tr(out)
    out = self.norm4bn_tr(out)
    out = MEF.relu(out)
    out = self.block4_tr(out)

    out = ME.cat(out, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3in_tr(out)
    out = self.norm3bn_tr(out)
    out = MEF.relu(out)
    out = self.block3_tr(out)

    out = ME.cat(out, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2in_tr(out)
    out = self.norm2bn_tr(out)
    out = MEF.relu(out)
    out = self.block2_tr(out)

    out = ME.cat(out, out_s1)
    out = self.conv1_tr(out)
    out = MEF.relu(out)
    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / (torch.norm(out.F, p=2, dim=1, keepdim=True) + 1e-8),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out


class ResUNetINBN2G(ResUNetINBN2):
  CHANNELS = [None, 128, 128, 128, 128]
  TR_CHANNELS = [None, 128, 128, 128, 128]


class ResUNetINBN2GC(ResUNetINBN2G):
  REGION_TYPE = ME.RegionType.HYPERCROSS


class ResUNetINBN2GPG(ResUNetINBN2GP):
  CHANNELS = [None, 128, 128, 128, 128]
  TR_CHANNELS = [None, 128, 128, 128, 128]


class ResUNetINBN2GPGC(ResUNetINBN2GPG):
  REGION_TYPE = ME.RegionType.HYPERCROSS


class ResUNetINBN2GP2GC(ResUNetINBN2GPG):
  USE_GPOOL2 = True
  REGION_TYPE = ME.RegionType.HYPERCROSS


class ResUNetINBN2GPGx2C(ResUNetINBN2GPGC):
  DEPTHS = [None, 1, 2, 2, 2, 2, 2, 1, None]


class ResUNetINBN2Gx2(ResUNetINBN2G):
  DEPTHS = [None, 1, 2, 2, 2, 2, 2, 1, None]


class ResUNetINBN2Gx2C(ResUNetINBN2Gx2):
  REGION_TYPE = ME.RegionType.HYPERCROSS


class ResUNetAINBN2G(ResUNetBN2G):
  NORM_TYPE = 'AINBN'
  BLOCK_NORM_TYPE = 'AINBN'