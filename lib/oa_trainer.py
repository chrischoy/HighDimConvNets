import gc
import logging

import numpy as np
import sklearn.metrics as metrics
import torch

from baseline.model.oanet import OALoss, OANet
from lib.timer import AverageMeter, Timer
from lib.trainer import Trainer
from lib.util_2d import compute_symmetric_epipolar_residual, compute_angular_error


class OATrainer(Trainer):
  """OANet trainer"""

  def __init__(self, config, data_loader, val_data_loader=None):
    Trainer.__init__(self, config, data_loader, val_data_loader)
    self.loss = OALoss(config)

  def _initialize_model(self):
    model = OANet(self.config)
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

      # To Cuda
      for key in input_dict.keys():
        if type(input_dict[key]) == torch.Tensor:
          input_dict[key] = input_dict[key].to(self.device)

      # Feature extraction
      inlier_timer.tic()
      norm_coords = input_dict['norm_coords'].unsqueeze(1)
      logits, e_hat = self.model(norm_coords)
      inlier_timer.toc()

      # Calculate loss
      labels = input_dict['labels']
      loss = 0
      for i in range(len(logits)):
        loss_i = self.loss.run(start_iter + curr_iter, input_dict, logits[i], e_hat[i])
        loss += loss_i
      loss.backward()
      self.optimizer.step()
      total_timer.toc()

      # Accumulate metrics
      pred = np.hstack(torch.sigmoid(logits[-1]).squeeze().detach().cpu().numpy())
      target = np.hstack(labels.squeeze().detach().cpu().numpy()).astype(np.int)
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
        # Use the current value to see how stochastic the metrics are
        stat = {
            'prec': prec_meter.val,
            'recall': recall_meter.val,
            'f1': f1_meter.val,
            'ap': ap_meter.val,
            'loss': loss_meter.val
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

        # To Cuda
        for key in input_dict.keys():
          if type(input_dict[key]) == torch.Tensor:
            input_dict[key] = input_dict[key].to(self.device)

        # Feature extraction
        inlier_timer.tic()
        norm_coords = input_dict['norm_coords'].unsqueeze(1)
        logits, e_hats = self.model(norm_coords)
        inlier_timer.toc()

        # Calculate loss
        labels = input_dict['labels']
        loss = 0
        for i in range(len(logits)):
          loss_i = self.loss.run(curr_iter, input_dict, logits[i], e_hats[i])
          loss += loss_i
        total_timer.toc()

        # Accumulate metrics
        pred = np.hstack(torch.sigmoid(logits[-1]).squeeze().cpu().numpy())
        target = np.hstack(labels.squeeze().cpu().numpy()).astype(np.int)
        prec, recall, f1, _ = metrics.precision_recall_fscore_support(
            target, (pred > 0.5).astype(np.int), average='binary')
        ap = metrics.average_precision_score(target, pred)

        prec_meter.update(prec)
        recall_meter.update(recall)
        f1_meter.update(f1)
        ap_meter.update(ap)
        loss_meter.update(loss.item())

        # Clean
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
        logits, e_hats = self.model(norm_coords.unsqueeze(1).to(self.device))
        logit = logits[-1].squeeze().cpu()
        e_hat = e_hats[-1].cpu().numpy()

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
