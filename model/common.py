import torch
import numpy as np
import MinkowskiEngine as ME


class AIN(torch.nn.Module):
  # Attentive Instance Normalization

  def __init__(self, num_feats):
    super(AIN, self).__init__()
    self.num_feats = num_feats
    self.local_linear = torch.nn.Linear(num_feats, 1)
    self.global_linear = torch.nn.Linear(num_feats, 1)

  def forward(self, x):
    feats = x.feats
    local_w = self.local_linear(feats)
    global_w = self.global_linear(feats)
    weight = torch.zeros_like(local_w)
    for row_idx in x.coords_man.get_row_indices_per_batch(x.coords_key):
      _local_w = local_w[row_idx]
      _local_w = torch.sigmoid(_local_w)
      _global_w = global_w[row_idx]
      _global_w = torch.softmax(_global_w, dim=0)
      weight[row_idx] = _local_w * _global_w

    # normalize weight
    weight = weight / torch.sum(torch.abs(weight))
    mean = torch.sum(feats * weight, dim=0, keepdim=True) / torch.sum(weight)
    std = torch.sqrt(torch.sum(weight*(feats - mean).pow(2), dim=0, keepdim=True))
    return ME.SparseTensor(
        feats=(feats - mean) / std,
        coords_key=x.coords_key,
        coords_manager=x.coords_man,
    )


def get_norm(norm_type, num_feats, bn_momentum=0.05, dimension=-1):
  if norm_type == 'BN':
    return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
  elif norm_type == 'IN':
    # return ME.MinkowskiInstanceNorm(num_feats, dimension=dimension)
    return ME.MinkowskiInstanceNorm(num_feats)
  elif norm_type == 'INBN':
    return torch.nn.Sequential(
        ME.MinkowskiInstanceNorm(num_feats),
        ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum))
  elif norm_type == 'AIN':
    return AIN(num_feats)
  elif norm_type == 'AINBN':
    return torch.nn.Sequential(
        AIN(num_feats), ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum))
  else:
    raise ValueError(f'Type {norm_type}, not defined')


def get_nonlinearity(non_type):
  if non_type == 'ReLU':
    return ME.MinkowskiReLU()
  elif non_type == 'ELU':
    # return ME.MinkowskiInstanceNorm(num_feats, dimension=dimension)
    return ME.MinkowskiELU()
  else:
    raise ValueError(f'Type {non_type}, not defined')


def random_offsets(kernel_size, n_kernel, dimension):
  n = kernel_size**dimension
  offsets = np.random.choice(n, n_kernel, replace=False)
  offsets = np.unravel_index(offsets, [
      kernel_size,
  ] * dimension)
  offsets = np.stack(offsets).T
  offsets = offsets - kernel_size // 2
  return offsets


def conv(in_channels,
         out_channels,
         kernel_size,
         stride=1,
         dilation=1,
         has_bias=False,
         region_type=ME.RegionType.HYPERCUBE,
         num_kernels=-1,
         dimension=-1):
  assert dimension > 0, 'Dimension must be a positive integer'
  if num_kernels > 0:
    offsets = random_offsets(kernel_size, num_kernels, dimension)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=ME.RegionType.CUSTOM,
        region_offsets=torch.IntTensor(offsets),
        dimension=dimension)
  else:
    kernel_generator = ME.KernelGenerator(
        kernel_size, stride, dilation, region_type=region_type, dimension=dimension)

  return ME.MinkowskiConvolution(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      has_bias=has_bias,
      kernel_generator=kernel_generator,
      dimension=dimension)


def conv_tr(in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            has_bias=False,
            region_type=ME.RegionType.HYPERCUBE,
            num_kernels=-1,
            dimension=-1):
  assert dimension > 0, 'Dimension must be a positive integer'
  if num_kernels > 0:
    offsets = random_offsets(kernel_size, num_kernels, dimension)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        is_transpose=True,
        region_type=ME.RegionType.CUSTOM,
        region_offsets=torch.IntTensor(offsets),
        dimension=dimension)
  else:
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        is_transpose=True,
        region_type=region_type,
        dimension=dimension)

  kernel_generator = ME.KernelGenerator(
      kernel_size,
      stride,
      dilation,
      is_transpose=True,
      region_type=region_type,
      dimension=dimension)

  return ME.MinkowskiConvolutionTranspose(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      has_bias=has_bias,
      kernel_generator=kernel_generator,
      dimension=dimension)


def conv_norm_non(inc,
                  outc,
                  kernel_size,
                  stride,
                  dimension,
                  bn_momentum=0.05,
                  region_type=ME.RegionType.HYPERCUBE,
                  norm_type='BN',
                  nonlinearity='ELU'):
  return torch.nn.Sequential(
      conv(
          in_channels=inc,
          out_channels=outc,
          kernel_size=kernel_size,
          stride=stride,
          dilation=1,
          has_bias=False,
          region_type=region_type,
          dimension=dimension),
      get_norm(norm_type, outc, bn_momentum=bn_momentum, dimension=dimension),
      get_nonlinearity(nonlinearity))
