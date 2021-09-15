import logging
import os.path as osp
import sys

import numpy as np
import open3d
import pandas as pd
import torch
from easydict import EasyDict as edict
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support)

from config import get_parser
from lib.twodim_data_loaders import make_data_loader
from lib.util import ensure_dir, read_txt
from train import get_trainer


def print_table(scenes, keys, values, out_dir, filename):
  data = dict()
  metrics = list(zip(*values))
  for k, metric in zip(keys, metrics):
    data[k] = metric

  df = pd.DataFrame(data, index=scenes)
  df.loc['mean'] = df.mean()
  print(df.to_string())
  df.to_csv(osp.join(out_dir, filename))


def load_scenes(config):
  dataset = config.dataset
  
  if 'YFCC100M' in dataset:
    scene_path = 'config/test_yfcc.txt'
  elif dataset == 'ThreeDMatchPairDataset':
    scene_path = 'config/test_3dmatch.txt'
  elif dataset == 'SUN3DDatasetExtracted':
    scene_path = 'config/test_sun3d.txt'
  else:
    raise ValueError(f"{dataset} is not supported")

  scene_list = read_txt(scene_path)
  return scene_list


def exp_prec_recall(target_list, pred_list, residual_list, scene_list, out_dir):
  logging.info("Exp 1. Evaluating classification scores")

  target_list = [np.hstack(targets) for targets in target_list]
  pred_list = [np.hstack(preds) for preds in pred_list]
  residual_list = [np.hstack(residuals) for residuals in residual_list]

  keys = ['prec', 'recall', 'f1', 'ap', 'mean', 'median']
  metrics = []
  for targets, preds, residuals in zip(target_list, pred_list, residual_list):
    prec, recall, f1, _ = precision_recall_fscore_support(
        targets, (preds > 0.5).astype(np.int), average='binary')
    ap = average_precision_score(targets, preds)
    mean = np.mean(residuals)
    median = np.median(residuals)
    metrics.append([prec, recall, f1, ap, mean, median])

  logging.info("Classification Scores")
  print_table(scene_list, keys, metrics, out_dir, 'prec_recall.csv')


def exp_ap_curve(target_list, pred_list, out_dir):
  logging.info(f"Exp 2. Drawing Prec-Recall curve")
  
  targets = np.hstack([np.hstack(targets) for targets in target_list])
  preds = np.hstack([np.hstack(preds) for preds in pred_list])

  prec, recall, _ = precision_recall_curve(targets, preds)
  idx = np.linspace(0, len(recall) - 1, 100).astype(np.int)
  prec = prec[idx]
  recall = recall[idx]
  np.savez(osp.join(out_dir, 'ap_curve.npz'), prec=prec, recall=recall)


def exp_distance_ap(residual_list, scene_list, out_dir):
  logging.info(f"Exp 3. Evaulating distance AP")

  ths = np.arange(20) * 0.01

  metrics = []
  for residuals in residual_list:
    res_acc_hist, _ = np.histogram(residuals, ths)
    res_acc_hist = res_acc_hist.astype(np.float) / float(residuals.shape[0])
    res_acc = np.cumsum(res_acc_hist)
    ap_list = [np.mean(res_acc[:i]) for i in range(1, len(ths))]
    metrics.append(ap_list)

  logging.info("mAP - Epipolar Distance")
  print_table(scene_list, ths[1:], metrics, out_dir, 'dist_ap.csv')


def exp_angular_ap(err_q_list, err_t_list, scene_list, out_dir):
  logging.info("Exp 4. Evaluating augular AP")

  num_ths = 7
  ths = np.arange(num_ths) * 5

  metric_q, metric_t, metric_qt = [], [], []
  for err_q, err_t in zip(err_q_list, err_t_list):
    q_acc_hist, _ = np.histogram(err_q, ths)
    q_acc_hist = q_acc_hist.astype(np.float) / float(err_q.shape[0])
    q_acc = np.cumsum(q_acc_hist)
    q_ap = [np.mean(q_acc[:i]) for i in range(1, len(ths))]
    metric_q.append(q_ap)

    t_acc_hist, _ = np.histogram(err_t, ths)
    t_acc_hist = t_acc_hist.astype(np.float) / float(err_t.shape[0])
    t_acc = np.cumsum(t_acc_hist)
    t_ap = [np.mean(t_acc[:i]) for i in range(1, len(ths))]
    metric_t.append(t_ap)

    qt_acc_hist, _ = np.histogram(np.maximum(err_q, err_t), ths)
    qt_acc_hist = qt_acc_hist.astype(np.float) / float(err_q.shape[0])
    qt_acc = np.cumsum(qt_acc_hist)
    qt_ap = [np.mean(qt_acc[:i]) for i in range(1, len(ths))]
    metric_qt.append(qt_ap)

  logging.info("mAP - Rotation")
  print_table(scene_list, ths[1:], metric_q, out_dir, 'q_ap.csv')

  logging.info("mAP - Translation")
  print_table(scene_list, ths[1:], metric_t, out_dir, 't_ap.csv')

  logging.info("mAP - Rotation & Translation")
  print_table(scene_list, ths[1:], metric_qt, out_dir, 'qt_ap.csv')


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

  if args.do_extract:
    Trainer = get_trainer(config.trainer)
    model = Trainer(config, [])

    target_list, pred_list, residual_list = [], [], []
    for scene in scenes:
      test_loader = make_data_loader(
          config,
          'test',
          batch_size=1,
          num_workers=1,
          shuffle=False,
          repeat=False,
          scene=scene)

      targets, preds, residuals, err_qs, err_ts = model.test(test_loader)
      mean_residuals = [np.mean(res) for res in residuals]
      err_qs = np.hstack(err_qs)
      err_ts = np.hstack(err_ts)

      logging.info(f"Save raw data - {scene}")
      np.savez(
          osp.join(args.out_dir, f"{scene}_raw"),
          targets=targets,
          preds=preds,
          residuals=residuals,
          mean_residuals=mean_residuals,
          err_qs=err_qs,
          err_ts=err_ts)

  target_list, pred_list, residual_list, mean_residual_list, err_q_list, err_t_list = [], [], [], [], [], []
  for scene in scenes:
    logging.info(f"Load raw data - {scene}")
    data = np.load(osp.join(args.out_dir, f"{scene}_raw.npz"), allow_pickle=True)
    target_list.append(data['targets'])
    pred_list.append(data['preds'])
    residual_list.append(data['residuals'])
    mean_residual_list.append(data['mean_residuals'])
    err_q_list.append(data['err_qs'])
    err_t_list.append(data['err_ts'])

  exp_prec_recall(target_list, pred_list, residual_list, scenes, args.out_dir)
  exp_ap_curve(target_list, pred_list, args.out_dir)
  exp_distance_ap(mean_residual_list, scenes, args.out_dir)
  exp_angular_ap(err_q_list, err_t_list, scenes, args.out_dir)
