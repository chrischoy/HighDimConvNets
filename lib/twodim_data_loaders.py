import logging
import os.path as osp

import cv2
import h5py
import MinkowskiEngine as ME
import numpy as np
import torch
import torch.utils.data

from lib.util_data import InfSampler
from lib.util_2d import compute_symmetric_epipolar_residual


class CollationFunctionFactory:
  def __init__(self, config):
    self.config = config
    if config.collation_2d == 'default':
      self.fn = self.collate
    elif config.collation_2d == 'collate_correspondence':
      self.fn = self.collate_correspondence
    elif config.collation_2d == 'collate_lfgc':
      self.fn = self.collate_oa
    elif config.collation_2d == 'collate_oa':
      self.fn = self.collate_oa
    else:
      raise ValueError(f'Invalid collation_2d {config.collation_2d}')

  def collate(self, data):
    if isinstance(data[0], dict):
      return {
          # "img0": [b["img0"] for b in data],
          # "img1": [b["img1"] for b in data],
          "coords": [b["coords"] for b in data],
          "labels": [b["labels"] for b in data],
          "norm_coords": [b["norm_coords"] for b in data],
          "E": [b["E"] for b in data]
      }
    return list(zip(*data))

  def collate_oa(self, batch):
    assert isinstance(batch[0], dict)

    img0_batch = [b['img0'] for b in batch]
    img1_batch = [b['img1'] for b in batch]
    coords_batch = [b['coords'] for b in batch]
    norm_coords_batch = [b['norm_coords'] for b in batch]
    labels_batch = [b['labels'] for b in batch]
    e_batch = [b['E'] for b in batch]
    virt_batch = [b['virtPts'] for b in batch]
    R_batch = [b['R'] for b in batch]
    t_batch = [b['t'] for b in batch]

    numkps = [coords.shape[0] for coords in norm_coords_batch]
    minkps = np.min(numkps)

    norm_coords_batch = [coords[:minkps, :] for coords in norm_coords_batch]
    labels_batch = [labels[:minkps] for labels in labels_batch]

    return {
        'coords': coords_batch,
        'norm_coords': torch.from_numpy(np.asarray(norm_coords_batch)),
        'labels': torch.from_numpy(np.asarray(labels_batch)),
        'E': torch.from_numpy(np.asarray(e_batch)),
        'R': torch.from_numpy(np.asarray(R_batch)),
        't': torch.from_numpy(np.asarray(t_batch)),
        'virtPts': torch.from_numpy(np.asarray(virt_batch)),
        'img0': img0_batch,
        'img1': img1_batch
    }

  def collate_correspondence(self, batch):
    assert isinstance(batch[0], dict)

    img0_batch = [b['img0'] for b in batch]
    img1_batch = [b['img1'] for b in batch]
    coords_batch = [b['coords'] for b in batch]
    norm_coords_batch = [b['norm_coords'] for b in batch]
    labels_batch = [b['labels'] for b in batch]
    E_batch = [b['E'] for b in batch]
    R_batch = [b['R'] for b in batch]
    t_batch = [b['t'] for b in batch]
    virt_batch = [b['virtPts'] for b in batch]

    sinput_C, sinput_F, sinput_L, idx_batch = [], [], [], []
    for norm_coords, labels in zip(norm_coords_batch, labels_batch):
      # quantize
      quan_coords = np.floor(norm_coords / self.config.quantization_size)
      idx, idx_reverse = ME.utils.sparse_quantize(quan_coords,
                                                  return_index=True,
                                                  return_inverse=True)
      C = quan_coords[idx]
      F = norm_coords[idx]
      L = labels[idx]

      sinput_C.append(C)
      sinput_F.append(F)
      sinput_L.append(L)
      idx_batch.append(idx)

    # Sample minimum number of coordinates for each batch
    if self.config.sample_minimum_coords:
      npts = [C.shape[0] for C in sinput_C]
      min_pts = np.min(npts)
      for i, (C, F, L) in enumerate(zip(sinput_C, sinput_F, sinput_L)):
        if C.shape[0] > min_pts:
          rand_idx = np.random.choice(C.shape[0], min_pts, replace=False)
          sinput_C[i] = C[rand_idx]
          sinput_F[i] = F[rand_idx]
          sinput_L[i] = L[rand_idx]
          idx_batch[i] = idx_batch[i][rand_idx]

    # Collate
    len_batch = [C.shape[0] for C in sinput_C]
    E_batch = torch.from_numpy(np.asarray(E_batch))
    R_batch = torch.from_numpy(np.asarray(R_batch))
    t_batch = torch.from_numpy(np.asarray(t_batch))
    virt_batch = torch.from_numpy(np.asarray(virt_batch))
    norm_coords_batch = sinput_F

    sinput_C_th = ME.utils.batched_coordinates(sinput_C)
    sinput_F_th = torch.from_numpy(np.vstack(sinput_F))
    sinput_L_th = torch.from_numpy(np.hstack(sinput_L))

    xyz = sinput_F_th

    # if inlier_feature_type is not coords, use ones as feature
    if self.config.inlier_feature_type != 'coords':
      sinput_F_th = torch.ones((len(sinput_C), 1))

    return {
        'sinput_C': sinput_C_th.int(),
        'sinput_F': sinput_F_th.float(),
        'sinput_L': sinput_L_th.int(),
        'virtPts': virt_batch.float(),
        'E': E_batch.float(),
        'R': R_batch.float(),
        't': t_batch.float(),
        'len_batch': len_batch,
        'xyz': xyz,
        'coords': coords_batch,
        'norm_coords': norm_coords_batch,
        'labels': labels_batch,
        'img0': img0_batch,
        'img1': img1_batch,
    }

  def __call__(self, data):
    return self.fn(data)


class YFCC100MDatasetExtracted(torch.utils.data.Dataset):
  DATA_FILES = {
      'train': 'yfcc-sift-2000-train.h5',
      'val': 'yfcc-sift-2000-val.h5',
      'test': 'yfcc-sift-2000-test.h5'
  }

  def __init__(self, phase, manual_seed, config, scene=None):
    self.phase = phase
    self.manual_seed = manual_seed
    self.config = config
    self.scene = scene

    # self.source_dir = config.data_dir_raw
    self.target_dir = config.data_dir_processed
    self.inlier_threshold_pixel = config.inlier_threshold_pixel

    self.data = None
    config_name = self.DATA_FILES[phase]
    if phase == 'test' and scene is not None:
      splits = config_name.split('.')
      config_name = splits[0] + f'-{scene}.' + splits[1]
    self.filename = osp.join(self.target_dir, config_name)
    logging.info(
        f"Loading {self.__class__.__name__} subset {phase} from {self.filename} with {self.__len__()} pairs."
    )

  def __len__(self):
    if self.data is None:
      self.data = h5py.File(self.filename, 'r')
      _len = len(self.data['coords'])
      self.data.close()
      self.data = None
    else:
      _len = len(self.data['coords'])
    return _len

  def __del__(self):
    if self.data is not None:
      self.data.close()

  def __getitem__(self, idx):
    if self.data is None:
      self.data = h5py.File(self.filename, 'r')

    idx = str(idx)
    coords = self.data['coords'][idx]
    # img_path0 = coords.attrs['img0']
    # img_path1 = coords.attrs['img1']
    coords = np.asarray(coords)
    norm_coords = np.asarray(self.data['n_coords'][idx])
    E = np.asarray(self.data['E'][idx])
    R = np.asarray(self.data['R'][idx])
    t = np.asarray(self.data['t'][idx])
    res = np.asarray(self.data['res'][idx])
    E = E / np.linalg.norm(E)

    # img0 = cv2.imread(osp.join(self.source_dir, img_path0))
    # img1 = cv2.imread(osp.join(self.source_dir, img_path1))
    img0 = 1
    img1 = 1

    labels = res < self.inlier_threshold_pixel
    virtPts = self.correctMatches(E)

    return {
        'img0': img0,
        'img1': img1,
        'coords': coords,
        'norm_coords': norm_coords,
        'labels': labels,
        'E': E,
        'R': R,
        't': t,
        'virtPts': virtPts,
    }

  def reset(self):
    if self.data is not None:
      self.data.close()
    self.data = None

  def correctMatches(self, E):
    step = 0.1
    xx, yy = np.meshgrid(np.arange(-1, 1, step), np.arange(-1, 1, step))
    # Points in first image before projection
    pts1_virt_b = np.float32(np.vstack((xx.flatten(), yy.flatten())).T)
    # Points in second image before projection
    pts2_virt_b = np.float32(pts1_virt_b)
    pts1_virt_b, pts2_virt_b = pts1_virt_b.reshape(1, -1,
                                                   2), pts2_virt_b.reshape(1, -1, 2)

    pts1_virt_b, pts2_virt_b = cv2.correctMatches(E.reshape(3, 3), pts1_virt_b,
                                                  pts2_virt_b)
    pts1_virt_b = pts1_virt_b.squeeze()
    pts2_virt_b = pts2_virt_b.squeeze()
    pts_virt = np.concatenate([pts1_virt_b, pts2_virt_b], axis=1)
    return pts_virt


class YFCC100MDatasetUCN(YFCC100MDatasetExtracted):
  DATA_FILES = {
      'train': 'yfcc-ucn-0-train.h5',
      'val': 'yfcc-ucn-0-val.h5',
      'test': 'yfcc-ucn-0-test.h5'
  }

  def __getitem__(self, idx):
    if self.data is None:
      self.data = h5py.File(self.filename, 'r')

    idx = str(idx)
    E = np.asarray(self.data['E'][idx])
    E = E / np.linalg.norm(E)
    R = np.asarray(self.data['R'][idx])
    t = np.asarray(self.data['t'][idx])
    img0 = 1
    img1 = 1

    # if self.phase != 'train':
    coords = np.asarray(self.data['ucn_coords'][idx])
    norm_coords = np.asarray(self.data['ucn_n_coords'][idx])
    res = compute_symmetric_epipolar_residual(E, norm_coords[:, :2], norm_coords[:, 2:])
    # else:
    # coords = np.asarray(self.data['coords'][idx])
    # norm_coords = np.asarray(self.data['n_coords'][idx])
    # res = np.asarray(self.data['res'][idx])

    labels = res < self.inlier_threshold_pixel
    virtPts = self.correctMatches(E)

    return {
        'img0': img0,
        'img1': img1,
        'coords': coords,
        'norm_coords': norm_coords,
        'labels': labels,
        'E': E,
        'R': R,
        't': t,
        'virtPts': virtPts,
    }


ALL_DATASETS = [YFCC100MDatasetExtracted, YFCC100MDatasetUCN]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config,
                     phase,
                     batch_size,
                     num_workers=0,
                     shuffle=None,
                     repeat=False,
                     scene=None):
  if config.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.dataset]

  dataset = Dataset(phase=phase, manual_seed=None, config=config, scene=scene)

  collate_fn = CollationFunctionFactory(config)

  if repeat:
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              sampler=InfSampler(dataset, shuffle))
  else:
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              shuffle=shuffle)

  return data_loader
