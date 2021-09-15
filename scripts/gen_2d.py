import argparse
import glob
import itertools
import logging
import os
import os.path as osp
import pickle
import sys

import cv2
from easydict import EasyDict as edict
import h5py
import numpy as np
import open3d
import torch
from tqdm import tqdm

from lib.eval import find_nn_gpu
import lib.util_2d as util_2d
from lib.util import random_sample, read_txt
from ucn.resunet import ResUNetBN2D2
from util.file import loadh5

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])


class SIFT(object):
  def __init__(self, config):
    self.num_kp = config.num_kp
    self.contrastThreshold = 1e-5
    self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.num_kp,
                                            contrastThreshold=self.contrastThreshold)

  def extract_and_save(self, path):
    assert osp.exists(path), 'file not exist'

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img.size > 0, 'empty images'

    kp, desc = self.sift.detectAndCompute(img, None)
    kp = np.array([_kp.pt for _kp in kp])

    desc_name = f'sift-{self.num_kp}'
    target_path = f'{path}.{desc_name}.h5'
    with h5py.File(target_path, 'w') as fp:
      fp.create_dataset('kp', kp.shape, dtype=np.float32)
      fp.create_dataset('desc', desc.shape, dtype=np.float32)
      fp['kp'][:] = kp
      fp['desc'][:] = desc


def dump_feature(config):
  img_glob = osp.join(config.source, '*/*/images/*.jpg')
  imgs = glob.glob(img_glob)
  logging.info(f'grab {len(imgs)} images')

  sift = SIFT(config)

  for img_path in tqdm(imgs):
    sift.extract_and_save(img_path)


class OpenUCN(object):
  def __init__(self, config):
    assert osp.exists(
        config.ucn_weight), f"UCN weight {config.ucn_weight} does not exist"
    self.device = torch.device('cuda')
    checkpoint = torch.load(config.ucn_weight)
    model = ResUNetBN2D2(1, 64, normalize_feature=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    self.model = model.to(self.device)
    self.config = config
    self.nn_max_n = config.nn_max_n
    self.ucn_inlier_threshold_pixel = config.ucn_inlier_threshold_pixel

  def dump_correspondence(
      self,
      img0,
      img1,
      F0,
      F1,
      mode='gpu-all',
  ):
    use_stability_test = False
    use_cyclic_test = False
    keypoint = 'sift'
    if keypoint == 'sift':
      sift = cv2.xfeatures2d.SIFT_create(
          0,
          9,
          0.01,  # Smaller more keypoints, default 0.04
          100  # larger more keypoints, default 10
      )
      kp0 = sift.detect(img0, None)
      kp1 = sift.detect(img1, None)
      xy_kp0 = np.floor(np.array([k.pt for k in kp0]).T)
      xy_kp1 = np.floor(np.array([k.pt for k in kp1]).T)
      x0, y0 = xy_kp0[0], xy_kp0[1]
      x1, y1 = xy_kp1[0], xy_kp1[1]
    elif keypoint == 'all':
      x0, y0 = None, None
      x1, y1 = None, None

    H0, W0 = img0.shape
    H1, W1 = img1.shape

    if mode == 'gpu-all':
      nn_inds1 = find_nn_gpu(F0[:, y0, x0],
                             F1.view(F1.shape[0], -1),
                             nn_max_n=self.nn_max_n,
                             transposed=True)

      # Convert the index to coordinate: BxCxHxW
      xs1 = nn_inds1 % W1
      ys1 = nn_inds1 // W1
      if use_stability_test:
        # Stability test: check stable under perturbation
        noisex = 2 * (np.random.rand(len(xs1)) < 0.5) - 1
        noisey = 2 * (np.random.rand(len(ys1)) < 0.5) - 1
        xs1n = np.clip(xs1 + noisex, 0, W1 - 1)
        ys1n = np.clip(ys1 + noisey, 0, H1 - 1)
      else:
        xs1n = xs1
        ys1n = ys1

      if use_cyclic_test:
        # Test reciprocity
        nn_inds0 = find_nn_gpu(F1[:, ys1n, xs1n],
                               F0.view(F0.shape[0], -1),
                               nn_max_n=self.nn_max_n,
                               transposed=True)

        # Convert the index to coordinate: BxCxHxW
        xs0 = (nn_inds0 % W0)
        ys0 = (nn_inds0 // W0)

        # Test cyclic consistency
        dist_sq_nn = (x0 - xs0)**2 + (y0 - ys0)**2
        mask = dist_sq_nn < (self.ucn_inlier_threshold_pixel**2)

      else:
        xs0 = x0
        ys0 = y0
        mask = np.ones(len(x0)).astype(bool)

    elif mode == 'gpu-all-all':
      nn_inds1 = find_nn_gpu(F0.view(F0.shape[0], -1),
                             F1.view(F1.shape[0], -1),
                             nn_max_n=self.nn_max_n,
                             transposed=True)

      inds0 = np.arange(len(nn_inds1))
      x0 = inds0 % W0
      y0 = inds0 // W0

      xs1 = nn_inds1 % W1
      ys1 = nn_inds1 // W1

      if use_stability_test:
        # Stability test: check stable under perturbation
        noisex = 2 * (np.random.rand(len(xs1)) < 0.5) - 1
        noisey = 2 * (np.random.rand(len(ys1)) < 0.5) - 1
        xs1n = np.clip(xs1 + noisex, 0, W1 - 1)
        ys1n = np.clip(ys1 + noisey, 0, H1 - 1)
      else:
        xs1n = xs1
        ys1n = ys1

      # Test reciprocity
      nn_inds0 = find_nn_gpu(F1[:, ys1n, xs1n],
                             F0.view(F0.shape[0], -1),
                             nn_max_n=self.nn_max_n,
                             transposed=True)

      # Convert the index to coordinate: BxCxHxW
      xs0 = nn_inds0 % W0
      ys0 = nn_inds0 // W0

      # Filter out the points that fail the cycle consistency
      dist_sq_nn = (x0 - xs0)**2 + (y0 - ys0)**2
      mask = dist_sq_nn < (config.ucn_inlier_threshold_pixel**2)

    kp0 = np.stack((xs0[mask], ys0[mask]), axis=1).astype(np.float)
    kp1 = np.stack((xs1[mask], ys1[mask]), axis=1).astype(np.float)

    return kp0, kp1

  def extract(self, path0, path1):
    with torch.no_grad():
      img0 = self.prep_image(path0)
      img1 = self.prep_image(path1)

      F0 = self.model(self.to_normalized_torch(img0))
      F1 = self.model(self.to_normalized_torch(img1))
      return self.dump_correspondence(img0, img1, F0[0], F1[0], mode='gpu-all')

  def prep_image(self, path):
    assert osp.exists(path), f"File {path} does not exist."
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

  def to_normalized_torch(self, img):
    img = img.astype(np.float32) / 255 - 0.5
    return torch.from_numpy(img).to(self.device)[None, None, :, :]


class Dataset(object):
  def __init__(self, config, name, split, seqs, pair_num):
    self.config = config
    self.source_dir = config.source
    self.target_dir = config.target
    self.desc_name = f'{config.feature}-{config.num_kp}'
    self.name = name
    self.split = split
    self.seqs = seqs
    self.pair_num = pair_num
    if config.onthefly:
      self.feature_extractor = OpenUCN(config)

    config_name = f'{name}-{self.desc_name}-{split}'
    if split == 'test' and len(seqs) == 1:
      config_name = f'{config_name}-{seqs[0]}'

    self.dump_file = osp.join(self.target_dir, f'{config_name}.h5')
    logging.info(f"Dataset {name}-{split}")
    self.dump()

  def dump(self):
    logging.info(f"Dump dataset")
    for seq in self.seqs:
      seq_source = osp.join(self.source_dir, seq, self.split)
      seq_target = osp.join(self.target_dir, seq, self.desc_name, self.split)
      sequence = Sequence(config,
                          seq,
                          seq_source,
                          seq_target,
                          self.pair_num,
                          feature_extractor=self.feature_extractor)
      sequence.dump()
    self.collect()

  def collect(self):
    logging.info(f"Collect dataset")
    keys = ['coords', 'n_coords', 'E', 'R', 't', 'res', 'pairs', 'img']

    pair_idx = 0
    with h5py.File(self.dump_file, 'w') as ofp:
      data = {}
      for k in keys:
        data[k] = ofp.create_group(k)

      with tqdm(total=len(self.seqs)) as pbar:
        for seq in self.seqs:
          pbar.set_description(f'collecting - {seq}'.ljust(20))
          base_name = osp.join(self.target_dir, seq, self.desc_name, self.split)
          data_seq = {}
          for k in keys:
            p = osp.join(base_name, k) + '.pkl'
            with open(p, 'rb') as fp:
              data_seq[k] = pickle.load(fp)

          n = len(data_seq['coords'])
          for i in range(n):
            for k in keys[:-2]:
              data_item = data_seq[k][i]

              data_i = data[k].create_dataset(str(pair_idx),
                                              data_item.shape,
                                              dtype=np.float32)
              data_i[:] = data_item.astype(np.float32)
              if k == 'coords':
                idx0, idx1 = data_seq['pairs'][i]
                data_i.attrs['img0'] = data_seq['img'][idx0]
                data_i.attrs['idx0'] = idx0
                data_i.attrs['img1'] = data_seq['img'][idx1]
                data_i.attrs['idx1'] = idx1
            pair_idx += 1
          pbar.update(1)


class Sequence(object):
  def __init__(self,
               config,
               seq,
               data_dir,
               target_dir,
               pair_num,
               pair_name=None,
               feature_extractor=None):
    self.config = config
    self.seq = seq
    self.source_dir = config.source
    self.data_dir = data_dir
    self.target_dir = target_dir
    self.onthefly = config.onthefly
    self.feature_extractor = feature_extractor
    if not osp.exists(target_dir):
      os.makedirs(target_dir)
    self.feature = config.feature
    self.desc_name = f'{config.feature}-{config.num_kp}'
    logging.info(f'>> sequence {seq}')

    if not self.is_ready():
      self.load_data()

      if pair_name is None:
        self.build_pair(config.vis_threshold, pair_num)
      else:
        with open(pair_name, 'rb') as f:
          self.pairs = pickle.load(f)
      logging.info('>> pair lens' + str(len(self.pairs)))
    else:
      logging.info('>> done')

  def load_data(self):
    data_dir = self.data_dir

    img_list = read_txt(osp.join(data_dir, "images.txt"))
    calib_list = read_txt(osp.join(data_dir, "calibration.txt"))
    vis_list = read_txt(osp.join(data_dir, "visibility.txt"))

    calib = []
    vis = []

    loading_iter = list(zip(calib_list, vis_list))
    with tqdm(total=len(loading_iter)) as pbar:
      pbar.set_description(f"loading - {self.seq}".ljust(20))
      for calib_file, vis_file in loading_iter:
        calib += [util_2d.serialize_calibration(osp.join(data_dir, calib_file))]
        vis += [np.loadtxt(osp.join(data_dir, vis_file)).flatten().astype('float32')]
        pbar.update(1)

    self.img_path = [osp.join(data_dir, img_name) for img_name in img_list]
    self.rel_imgpath = [osp.relpath(p, self.source_dir) for p in self.img_path]
    self.calib = calib
    self.vis = vis

  def load_dumpfeature(self, imgpath):
    data = loadh5(f'{imgpath}.{self.desc_name}.h5')
    kp = data['kp']
    desc = data['desc']

    return kp, desc

  def build_pair(self, threshold, pair_num):
    n = len(self.vis)
    pairs = list(itertools.product(range(n), range(n)))

    # Filter pairs whose visibility is higher than threshold
    def check_visibility(d):
      i, j = d
      return i != j and self.vis[i][j] > threshold

    pairs = list(filter(check_visibility, pairs))
    pairs = random_sample(pairs, pair_num)
    self.pairs = pairs

  def dump_pair(self, i, j):
    assert self.onthefly and self.feature_extractor is not None, "feature extractor is not specified"

    img_path0, img_path1 = self.img_path[i], self.img_path[j]
    if not self.onthefly:
      kp1, desc1 = self.load_dumpfeature(img_path1)
      kp0, desc0 = self.load_dumpfeature(img_path0)
      matches = util_2d.computeNN(desc0, desc1)
      idx0, idx1 = matches
      kp0 = kp0[idx0]
      kp1 = kp1[idx1]
    else:
      # extract keypoints and features on the fly
      kp0, kp1 = self.feature_extractor.extract(img_path0, img_path1)

    calib0 = util_2d.parse_calibration(self.calib[i])
    calib1 = util_2d.parse_calibration(self.calib[j])
    T0 = util_2d.build_extrinsic_matrix(calib0['R'], calib0['t'])
    T1 = util_2d.build_extrinsic_matrix(calib1['R'], calib1['t'])
    E, R, t = util_2d.compute_essential_matrix(T0, T1)
    E = E / np.linalg.norm(E)

    n_kp0 = util_2d.normalize_keypoint(kp0, calib0['K'], calib0['imsize'] * 0.5)[:, :2]
    n_kp1 = util_2d.normalize_keypoint(kp1, calib1['K'], calib1['imsize'] * 0.5)[:, :2]

    coords = np.hstack((kp0, kp1))
    n_coords = np.hstack((n_kp0, n_kp1))

    residuals = util_2d.compute_symmetric_epipolar_residual(
        E,
        n_coords[:, :2],
        n_coords[:, 2:],
    )
    return coords, n_coords, E, R, t, residuals

  def dump(self):
    ready_file = osp.join(self.target_dir, 'ready')
    if not osp.exists(ready_file):
      res_dict = {}
      keys = ['coords', 'n_coords', 'E', 'R', 't', 'res']

      for k in keys:
        res_dict[k] = []

      with tqdm(total=len(self.pairs)) as pbar:
        pbar.set_description(f"dumping - {self.seq}".ljust(20))
        for i, j in self.pairs:
          res = self.dump_pair(i, j)
          for key_idx, key_name in enumerate(keys):
            res_dict[key_name] += [res[key_idx]]
          pbar.update(1)
      # Save
      for k in keys:
        output_file = osp.join(self.target_dir, k) + '.pkl'
        with open(output_file, 'wb') as ofp:
          pickle.dump(res_dict[k], ofp)
      with open(osp.join(self.target_dir, 'pairs') + '.pkl', 'wb') as ofp:
        pickle.dump(self.pairs, ofp)
      with open(osp.join(self.target_dir, 'img') + '.pkl', 'wb') as ofp:
        pickle.dump(self.rel_imgpath, ofp)

      with open(ready_file, 'w') as ofp:
        ofp.write('ready')

    logging.info('>> done')

  def is_ready(self):
    ready_file = osp.join(self.target_dir, 'ready')
    return osp.exists(ready_file)


if __name__ == "__main__":
  # Get config
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--source',
      type=str,
      required=True,
      help="source directory of YFCC100M dataset",
  )
  parser.add_argument(
      '--target',
      type=str,
      required=True,
      help="target directory to save processed data",
  )
  parser.add_argument('--vis_threshold',
                      type=float,
                      default=100,
                      help="visibility threshold to filter valid image pairs")
  parser.add_argument(
      '--num_kp',
      type=int,
      default=2000,
      help='number of features to extract',
  )
  parser.add_argument(
      '--nn_max_n',
      type=int,
      default=50,
      )
  parser.add_argument(
      '--dataset',
      type=str,
      default='yfcc',
      choices=['yfcc', 'sun3d'],
  )
  parser.add_argument('--feature', type=str, default='sift', choices=['sift', 'ucn'])
  parser.add_argument(
      '--onthefly',
      action='store_true',
  )
  parser.add_argument('--ucn_weight', type=str)
  parser.add_argument('--ucn_inlier_threshold_pixel', type=int, default=4)

  config = parser.parse_args()
  if config.feature == 'ucn':
    config = vars(config)
    config['num_kp'] = 0
    config = edict(config)

  source = config.source
  target = config.target
  dataset = config.dataset

  if not config.onthefly:
    dump_feature(config)

  test_seqs = read_txt(f'./config/test_{dataset}.txt')
  train_seqs = read_txt(f'./config/train_{dataset}.txt')

  for test_seq in test_seqs:
    test_dataset = Dataset(config, dataset, 'test', [test_seq], 1000)
  val_dataset = Dataset(config, dataset, 'val', train_seqs, 100)
  train_dataset = Dataset(config, dataset, 'train', train_seqs, 10000)
