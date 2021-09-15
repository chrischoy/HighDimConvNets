import gc
import logging

import numpy as np
import sklearn.metrics as metrics
import torch

from baseline.model.lfgc import LFGCLoss, LFGCNet
from lib.timer import AverageMeter, Timer
from lib.trainer import Trainer
from lib.util_2d import (compute_angular_error, compute_symmetric_epipolar_residual,
                         weighted_8points)


class LFGCTrainer(Trainer):
  """LFGC trainer"""

  def __init__(self, config, data_loader, val_data_loader=None):
    Trainer.__init__(self, config, data_loader, val_data_loader)
    self.loss = LFGCLoss(
        alpha=1.0, beta=0.1, regression_iter=config.regression_loss_iter)

  def _initialize_model(self):
    model = LFGCNet()
    return model

  def _train_epoch(self, epoch):
    gc.collect()

    data_loader_iter = self.data_loader.__iter__()
    loss_meter, prec_meter, recall_meter, f1_meter, ap_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, inlier_timer, total_timer = Timer(), Timer(), Timer()

    tot_num_data = len(data_loader_iter)
    if self.train_max_iter > 0:
      tot_num_data = min(self.train_max_iter, tot_num_data)
    start_iter = (epoch - 1) * tot_num_data

    self.model.train()
    for curr_iter in range(tot_num_data):
      self.optimizer.zero_grad()
      total_timer.tic()

      # Load data
      data_timer.tic()
      input_dict = data_loader_iter.next()
      data_timer.toc()

      # Feature extraction
      inlier_timer.tic()
      norm_coords = input_dict['norm_coords'].transpose(2, 1).to(self.device)
      logits = self.model(norm_coords).squeeze(1)
      inlier_timer.toc()

      # Calculate loss
      labels = input_dict['labels'].to(self.device)
      e = input_dict['E'].to(self.device)
      loss = self.loss(logits, norm_coords, labels, e, start_iter + curr_iter)
      loss.backward()

      # Check gradient explode
      explode = False
      for _, param in self.model.named_parameters():
        if torch.any(torch.isnan(param.grad)):
          explode = True

      if explode:
        total_timer.toc()
        continue

      self.optimizer.step()
      total_timer.toc()

      # Accumulate metrics
      pred = np.hstack(torch.sigmoid(logits).squeeze().detach().cpu().numpy())
      target = np.hstack(labels.cpu().numpy()).astype(np.int)
      prec, recall, f1, _ = metrics.precision_recall_fscore_support(
          target, (pred > 0.5).astype(np.int), average='binary')
      ap = metrics.average_precision_score(target, pred)

      prec_meter.update(prec)
      recall_meter.update(recall)
      f1_meter.update(f1)
      ap_meter.update(ap)
      loss_meter.update(loss.item())

      torch.cuda.empty_cache()

      if curr_iter % self.config.stat_freq == 0:
        stat = {
            'prec': prec_meter.avg,
            'recall': recall_meter.avg,
            'f1': f1_meter.avg,
            'ap': ap_meter.avg,
            'loss': loss_meter.avg
        }

        for k, v in stat.items():
          self.writer.add_scalar(f'train/{k}', v, start_iter + curr_iter)

        logging.info(
            ', '.join([f"Train Epoch: {epoch} [{curr_iter}/{tot_num_data}]"] +
                      [f"{k.capitalize()}: {v:.4f}" for k, v in stat.items()] + [
                          f"Data time: {data_timer.avg:.4f}",
                          f"Train time: {total_timer.avg - data_timer.avg:.4f}",
                          f"Total time: {total_timer.avg:.4f}"
                      ]))

  def _valid_epoch(self):
    gc.collect()

    data_loader_iter = self.val_data_loader.__iter__()
    loss_meter, prec_meter, recall_meter, f1_meter, ap_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, inlier_timer, total_timer = Timer(), Timer(), Timer()

    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)

    self.model.eval()
    with torch.no_grad():
      for curr_iter in range(tot_num_data):
        total_timer.tic()

        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_timer.toc()

        # Feature extraction
        inlier_timer.tic()
        norm_coords = input_dict['norm_coords'].transpose(2, 1).to(self.device)
        logits = self.model(norm_coords)
        logits = logits.squeeze(1)
        inlier_timer.toc()

        # Calculate loss
        labels = input_dict['labels'].to(self.device)
        e = input_dict['E'].to(self.device)
        loss = self.loss(logits, norm_coords, labels, e, curr_iter)
        total_timer.toc()

        # Accumulate metrics
        pred = np.hstack(torch.sigmoid(logits).squeeze().cpu().numpy())
        target = np.hstack(labels.cpu().numpy()).astype(np.int)
        prec, recall, f1, _ = metrics.precision_recall_fscore_support(
            target, (pred > 0.5).astype(np.int), average='binary')
        ap = metrics.average_precision_score(target, pred)

        prec_meter.update(prec)
        recall_meter.update(recall)
        f1_meter.update(f1)
        ap_meter.update(ap)
        loss_meter.update(loss.item())

        torch.cuda.empty_cache()

        if curr_iter % self.config.stat_freq == 0:
          stat = {
              'prec': prec_meter.avg,
              'recall': recall_meter.avg,
              'f1': f1_meter.avg,
              'ap': ap_meter.avg,
              'loss': loss_meter.avg
          }

          logging.info(
              ', '.join([f"Validation [{curr_iter}/{tot_num_data}]"] +
                        [f"{k.capitalize()}: {v:.4f}" for k, v in stat.items()] + [
                            f"Data time: {data_timer.avg:.4f}",
                            f"Train time: {total_timer.avg - data_timer.avg:.4f}",
                            f"Total time: {total_timer.avg:.4f}"
                        ]))

    stat = {
        'prec': prec_meter.avg,
        'recall': recall_meter.avg,
        'f1': f1_meter.avg,
        'ap': ap_meter.avg,
        'loss': loss_meter.avg
    }
    logging.info(', '.join([f"Validation"] +
                           [f"{k.capitalize()}: {v:.4f}" for k, v in stat.items()]))

    return stat

  def test(self, test_loader):
    test_iter = test_loader.__iter__()
    logging.info(f"Evaluating on {test_loader.dataset.scene}")

    self.model.eval()
    labels, preds, residuals, err_qs, err_ts = [], [], [], [], []
    with torch.no_grad():
      for _ in range(len(test_iter)):
        input_dict = test_iter.next()

        norm_coords = input_dict['norm_coords']

        coords_input = norm_coords.transpose(2, 1).to(self.device)
        logit = self.model(coords_input)
        logit = logit.squeeze(1)
        e_hat = weighted_8points(coords_input, logit)
        logit = logit.cpu()
        e_hat = e_hat.cpu().numpy()

        label = np.hstack(input_dict['labels'])
        pred = np.hstack(torch.sigmoid(logit))
        norm_coords = np.hstack(norm_coords)
        R = np.hstack(input_dict['R'])
        t = np.hstack(input_dict['t'])

        residual = compute_symmetric_epipolar_residual(
            e_hat.reshape(3, 3).T,
            norm_coords[label.astype(bool), :2],
            norm_coords[label.astype(bool), 2:],
        )

        err_q, err_t = compute_angular_error(
            R,
            t,
            e_hat.reshape(3, 3),
            norm_coords,
            pred,
        )

        labels.append(label.astype(np.int))
        preds.append(pred)
        residuals.append(residual)
        err_qs.append(err_q)
        err_ts.append(err_t)
    return labels, preds, residuals, err_qs, err_ts
