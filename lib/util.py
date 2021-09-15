import os
import torch
import copy
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path, mode=0o755)


def paint_overlap_label(pcd, overlap):
  npcd = np.asarray(pcd.points).shape[0]

  for i in range(npcd):
    if overlap[i] >= 1:
      pcd.colors[i] = [1.0, 0.0, 0.0]
  return pcd


def visualize_overlap_label(source, target, source_overlap, target_overlap, trans):
  source_temp = copy.deepcopy(source)
  target_temp = copy.deepcopy(target)
  source_temp.transform(trans)
  source_temp.paint_uniform_color([1, 0.706, 0])
  target_temp.paint_uniform_color([0, 0.651, 0.929])
  paint_overlap_label(source_temp, source_overlap)
  paint_overlap_label(target_temp, target_overlap)
  o3d.draw_geometries([source_temp, target_temp])


def extract_graph_features_from_batch(batch, features, i):
  g = torch.masked_select(features, (batch.batch == i).byte().unsqueeze(1).expand(
      -1, features.size(1)))
  g = g.view(-1, features.size(1))
  return g


def get_pointcloud_from_pytorch(batch, idx, R=None, T=None):
  p0 = o3d.PointCloud()
  pts = extract_graph_features_from_batch(batch, batch.x, 0).data.cpu().numpy()

  if R is not None:
    pts = (R.data.cpu().numpy() @ pts.T).T + T.data.cpu().numpy()

  p0.points = o3d.Vector3dVector(pts)

  return p0


def R_to_quad(R):
  q = torch.zeros(4)

  q[0] = 0.5 * ((1 + R[0, 0] + R[1, 1] + R[2, 2]).sqrt())
  q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
  q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
  q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])

  return q


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False):
  '''
  xyz is a N x 3 matrix
  rgb is a N x 3 matrix and all color must range from [0, 1] or None
  normal is a N x 3 matrix and all normal range from [-1, 1] or None

  if both rgb and normal are None, we use Nx1 one vector as an input

  if device is None, it tries to use gpu by default

  if skip_check is True, skip rigorous checks to speed up

  model = model.to(device)
  xyz, feats = extract_features(model, xyz)
  '''

  if not skip_check:
    assert xyz.shape[1] == 3

    N = xyz.shape[0]
    if rgb is not None:
      assert N == len(rgb)
      assert rgb.shape[1] == 3
      if np.any(rgb > 1):
        raise ValueError('Invalid color. Color must range from [0, 1]')

    if normal is not None:
      assert N == len(normal)
      assert normal.shape[1] == 3
      if np.any(normal > 1):
        raise ValueError('Invalid normal. Normal must range from [-1, 1]')

  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  feats = []
  if rgb is not None:
    # [0, 1]
    feats.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats.append(normal / 2)

  if rgb is None and normal is None:
    feats.append(np.ones((len(xyz), 1)))

  feats = np.hstack(feats)

  # Voxelize xyz and feats
  coords = np.floor(xyz / voxel_size)
  inds = ME.utils.sparse_quantize(coords, return_index=True)
  coords = coords[inds]
  # Append the batch index
  coords = np.hstack([coords, np.zeros((len(coords), 1))])
  return_coords = xyz[inds]

  feats = feats[inds]

  feats = torch.tensor(feats, dtype=torch.float32)
  coords = torch.tensor(coords, dtype=torch.int32)

  stensor = ME.SparseTensor(feats, coords=coords).to(device)

  return return_coords, model(stensor).F


def concat_pos_pairs(pos_pairs, len_batch):
  cat_pos_pairs = []
  start_inds = torch.zeros((1, 2)).long()
  assert len(pos_pairs) == len(len_batch)
  for pos_pair, lens in zip(pos_pairs, len_batch):
    cat_pos_pairs.append(pos_pair + start_inds)
    start_inds += torch.LongTensor(lens)
  return torch.cat(cat_pos_pairs, 0)


def random_sample(arr, num_sample, fix=True):
  """Sample elements from array

  Args:
    arr (array): array to sample
    num_sample (int): maximum number of elements to sample

  Returns:
    array: sampled array

  """
  # Fix seed
  if fix:
    np.random.seed(0)

  total = len(arr)
  num_sample = min(total, num_sample)
  idx = sorted(np.random.choice(range(total), num_sample, replace=False))
  return np.asarray(arr)[idx]


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)