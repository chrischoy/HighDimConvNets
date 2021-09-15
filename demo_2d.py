import itertools
import os.path as osp
from urllib.request import urlretrieve

import open3d as o3d
import cv2
import matplotlib.gridspec as grid
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch
import MinkowskiEngine as ME

from model import load_model
from lib.eval import find_nn_gpu
from lib.util import read_txt, ensure_dir
import lib.util_2d as util_2d
from ucn.resunet import ResUNetBN2D2
from util.file import loadh5

imgs = [
    '68833924_5994205213.jpg',
    '54990444_8865247484.jpg',
    '57895226_4857581382.jpg',
]
calibs = [
    'calibration_000002.h5',
    'calibration_000344.h5',
    'calibration_000489.h5',
]
output_dir = './visualize'

# downaload weights
if not osp.isfile('ResUNetBN2D2-YFCC100train.pth'):
  print("Downloading UCN weights...")
  urlretrieve(
      "https://node1.chrischoy.org/data/publications/ucn/ResUNetBN2D2-YFCC100train-100epoch.pth",
      'ResUNetBN2D2-YFCC100train.pth')

if not osp.isfile('2d_pyramid_ucn.pth'):
  print("Downloading PyramidSCNoBlock weights...")
  urlretrieve("http://cvlab.postech.ac.kr/research/hcngpr/data/2d_pyramid_ucn.pth",
              "2d_pyramid_ucn.pth")


def prep_image(full_path):
  assert osp.exists(full_path), f"File {full_path} does not exist."
  img = cv2.imread(full_path)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # return cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
  return img_gray, img_color


def to_normalized_torch(img, device):
  """
  Normalize the image to [-0.5, 0.5] range and augment batch and channel dimensions.
  """
  img = img.astype(np.float32) / 255 - 0.5
  return torch.from_numpy(img).to(device)[None, None, :, :]


def dump_correspondence(img0, img1, F0, F1, mode='gpu-all', pixel_ths=4):
  use_stability_test = True
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
                           nn_max_n=50,
                           transposed=True).numpy()

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
                             nn_max_n=50,
                             transposed=True).numpy()

      # Convert the index to coordinate: BxCxHxW
      xs0 = (nn_inds0 % W0)
      ys0 = (nn_inds0 // W0)

      # Test cyclic consistency
      dist_sq_nn = (x0 - xs0)**2 + (y0 - ys0)**2
      mask = dist_sq_nn < (pixel_ths**2)

    else:
      xs0 = x0
      ys0 = y0
      mask = np.ones(len(x0)).astype(bool)

  elif mode == 'gpu-all-all':
    nn_inds1 = find_nn_gpu(
        F0.view(F0.shape[0], -1),
        F1.view(F1.shape[0], -1),
        nn_max_n=50,
        transposed=True,
    ).numpy()

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
                           nn_max_n=50,
                           transposed=True).numpy()

    # Convert the index to coordinate: BxCxHxW
    xs0 = nn_inds0 % W0
    ys0 = nn_inds0 // W0

    # Filter out the points that fail the cycle consistency
    dist_sq_nn = (x0 - xs0)**2 + (y0 - ys0)**2
    mask = dist_sq_nn < (pixel_ths**2)

  return x0[mask], y0[mask], xs1[mask], ys1[mask]


def draw_figure(img0, img1, coords, labels, preds):
  plt.clf()
  fig = plt.figure()
  ratios = ratios = [img0.shape[1] * img1.shape[0], img1.shape[1] * img0.shape[0]]
  gs = grid.GridSpec(nrows=2, ncols=1, height_ratios=ratios)
  ax1 = fig.add_subplot(gs[0])
  ax2 = fig.add_subplot(gs[1])
  ax1.axis('off')
  ax2.axis('off')
  preds = preds > 0.5
  coords = coords[preds]
  labels = labels[preds]

  for coord, is_inlier in zip(coords, labels):
    con = ConnectionPatch(xyA=coord[:2],
                          xyB=coord[2:],
                          coordsA="data",
                          coordsB="data",
                          axesA=ax2,
                          axesB=ax1,
                          color="green" if is_inlier else "red")
    ax2.add_artist(con)

  ax1.imshow(img1)
  ax2.imshow(img0)
  plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
  return fig


def demo():
  root = './imgs'
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # load UCN model
  print(f"loading UCN model...")
  checkpoint = torch.load('ResUNetBN2D2-YFCC100train.pth')
  ucn = ResUNetBN2D2(1, 64, normalize_feature=True)
  ucn.load_state_dict(checkpoint['state_dict'])
  ucn.eval()
  ucn = ucn.to(device)

  # load HighDimConvNet
  print(f"loading HighDimConvNet model...")
  checkpoint = torch.load('2d_pyramid_ucn.pth')
  opts = checkpoint['config']
  Model = load_model(opts.inlier_model)
  model = Model(in_channels=4, out_channels=1, clusters=opts.oa_clusters, D=4)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()
  model = model.to(device)

  idx_list = itertools.combinations(range(len(imgs)), 2)
  with torch.no_grad():
    for i, (idx0, idx1) in enumerate(idx_list):
      # extract UCN features
      img0, img0_color = prep_image(osp.join(root, imgs[idx0]))
      img1, img1_color = prep_image(osp.join(root, imgs[idx1]))
      F0 = ucn(to_normalized_torch(img0, device))
      F1 = ucn(to_normalized_torch(img1, device))

      # load calibration data
      calib0 = loadh5(osp.join(root, calibs[idx0]))
      calib1 = loadh5(osp.join(root, calibs[idx1]))
      K0, K1 = calib0['K'], calib1['K']
      imsize0, imsize1 = calib0['imsize'], calib1['imsize']
      T0 = util_2d.build_extrinsic_matrix(calib0['R'], calib0['T'][0])
      T1 = util_2d.build_extrinsic_matrix(calib1['R'], calib1['T'][0])
      E, *_ = util_2d.compute_essential_matrix(T0, T1)
      E = E / np.linalg.norm(E)

      # dump correspondences
      x0, y0, x1, y1 = dump_correspondence(img0,
                                           img1,
                                           F0[0],
                                           F1[0],
                                           mode='gpu-all',
                                           pixel_ths=4)
      kp0 = np.stack((x0, y0), 1).astype(np.float)
      kp1 = np.stack((x1, y1), 1).astype(np.float)
      # normalize correspondence
      norm_kp0 = util_2d.normalize_keypoint(kp0, K0, imsize0 * 0.5)[:, :2]
      norm_kp1 = util_2d.normalize_keypoint(kp1, K1, imsize1 * 0.5)[:, :2]
      coords = np.concatenate((kp0, kp1), axis=1)
      norm_coords = np.concatenate((norm_kp0, norm_kp1), axis=1)

      # build HighDimConvNet input
      quan_coords = np.floor(norm_coords / opts.quantization_size)
      idx = ME.utils.sparse_quantize(quan_coords, return_index=True)
      C = quan_coords[idx]
      F = torch.from_numpy(norm_coords[idx]).float()
      C = ME.utils.batched_coordinates([C])
      sinput = ME.SparseTensor(coords=C, feats=F).to(device)
      input_dict = dict(xyz=F, len_batch=[len(F)])

      # feed forward
      logits, _ = model(sinput, input_dict)
      logits = logits[-1].squeeze().cpu()
      preds = np.hstack(torch.sigmoid(logits))
      residuals = util_2d.compute_symmetric_epipolar_residual(
          E, norm_coords[:, :2], norm_coords[:, 2:])
      labels = residuals < 0.01

      # draw figure
      fig = draw_figure(img0_color, img1_color, coords[idx], labels[idx], preds)
      filename = osp.join(output_dir, f"demo_{i}.png")
      fig.savefig(filename, bbox_inches='tight')
      print(f"save {filename}")


if __name__ == "__main__":
  ensure_dir(output_dir)
  demo()
