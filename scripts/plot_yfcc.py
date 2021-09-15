import logging
import os.path as osp
import sys

import cv2
import matplotlib.gridspec as grid
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import open3d
import pandas as pd
import torch
from easydict import EasyDict as edict

from config import get_parser
from lib.twodim_data_loaders import make_data_loader
from lib.util import ensure_dir, read_txt
from train import get_trainer
from scripts.benchmark_yfcc import load_scenes


def draw_figure(img0, img1, coords, labels, preds):
  # prepare figure
  plt.clf()
  fig = plt.figure()
  ratios = [img0.shape[1] * img1.shape[0], img1.shape[1] * img0.shape[0]]
  gs = grid.GridSpec(nrows=2, ncols=1, height_ratios=ratios)
  ax1 = fig.add_subplot(gs[0])
  ax2 = fig.add_subplot(gs[1])
  ax1.axis('off')
  ax2.axis('off')
  preds = preds > 0.5
  coords = coords[preds]
  labels = labels[preds]

  img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
  for coord, is_inlier in zip(coords, labels):
    con = ConnectionPatch(
        xyA=coord[:2],
        xyB=coord[ 2:],
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


if __name__ == "__main__":
  # args
  parser = get_parser()
  parser.add_argument(
      '--do_extract',
      action='store_true',
      help='extract network output by feed-forwarding data')

  args = parser.parse_args()
  ensure_dir(args.out_dir)

  # setup logger
  ch = logging.StreamHandler(sys.stdout)
  logging.getLogger().setLevel(logging.INFO)
  logging.basicConfig(
      format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

  logging.info("Start Benchmark")

  # prepare model
  checkpoint = torch.load(args.weights)

  config = checkpoint['config']
  config.data_dir_raw = args.data_dir_raw
  config.data_dir_processed = args.data_dir_processed
  config.weights = args.weights
  config.out_dir = args.out_dir
  config.resume = None

  vargs = vars(args)
  for k, v in config.items():
    vargs[k] = v
  config = edict(vargs)

  scenes = load_scenes(config)

  for scene in scenes:
    logging.info(f"Load raw data - {scene}")
    data = np.load(osp.join(args.out_dir, f"{scene}_raw.npz"), allow_pickle=True)
    preds = data['preds']

    figure_dir = osp.join(args.out_dir, "figures")
    ensure_dir(figure_dir)

    test_loader = make_data_loader(
        config,
        'test',
        batch_size=1,
        num_workers=1,
        shuffle=False,
        repeat=False,
        scene=scene)
    test_iter = test_loader.__iter__()

    for i in range(len(test_iter)):
      input_dict = test_iter.next()
      fig = draw_figure(
          img0=input_dict['img0'][0],
          img1=input_dict['img1'][0],
          coords=input_dict['coords'][0],
          labels=input_dict['labels'][0],
          preds=preds[i],
      )
      filename = osp.join(figure_dir, f"{scene[0]}{i:03d}.png")
      fig.savefig(filename, dpi=100, bbox_inches='tight')
      logging.info(f"save {filename}")
      plt.close(fig)
