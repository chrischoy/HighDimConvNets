from abc import abstractmethod
import gc
import logging

import MinkowskiEngine as ME
import numpy as np
import sklearn.metrics as metrics
import torch

from lib.timer import AverageMeter, Timer
from lib.trainer import Trainer
from lib.util_2d import (batch_episym, compute_e_hat,
                         compute_symmetric_epipolar_residual, compute_angular_error)
from model import load_model


class Loss(object):

  def __init__(self, config, device):
    self.config = config
    self.device = device
    self.loss_essential = config.oa_loss_essential
    self.loss_classif = config.oa_loss_classif
    self.geo_loss_margin = config.oa_geo_loss_margin
    self.loss_essential_init_iter = config.oa_loss_essential_init_iter
    self.use_balance_loss = config.use_balance_loss
    self.bce = torch.nn.BCEWithLogitsLoss()

  def run(self, step, data, logits, e_hats):
    labels, pts_virt = data['sinput_L'], data['virtPts']

    loss_e = self.essential_loss(logits, e_hats, pts_virt)
    loss_c = self.classif_loss(labels, logits)
    loss = 0

    # Check global_step and add essential loss
    if self.loss_essential > 0 and step >= self.loss_essential_init_iter:
      loss += self.loss_essential * loss_e
    if self.loss_classif > 0:
      loss += self.loss_classif * loss_c

    return loss

  def essential_loss(self, logits, e_hat, pts_virt):
    e_hat = e_hat.to(self.device)
    pts_virt = pts_virt.to(self.device)

    p1 = pts_virt[:, :, :2]
    p2 = pts_virt[:, :, 2:]
    geod = batch_episym(p1, p2, e_hat)
    loss = torch.min(geod, self.geo_loss_margin * geod.new_ones(geod.shape))
    loss = loss.mean()
    return loss

  def classif_loss(self, labels, logits):
    is_pos = labels.to(device=self.device, dtype=torch.bool)
    is_neg = ~is_pos
    is_pos = is_pos.to(logits.dtype)
    is_neg = is_neg.to(logits.dtype)

    if self.use_balance_loss:
      c = is_pos - is_neg
      loss = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())
      num_pos = torch.relu(torch.sum(is_pos, dim=0) - 1.0) + 1.0
      num_neg = torch.relu(torch.sum(is_neg, dim=0) - 1.0) + 1.0
      loss_p = torch.sum(loss * is_pos, dim=0)
      loss_n = torch.sum(loss * is_neg, dim=0)
      loss = torch.mean(loss_p * 0.5 / num_pos + loss_n * 0.5 / num_neg)
    else:
      loss = self.bce(logits, is_pos)
    return loss


class ImageCorrespondenceTrainer(Trainer):

  def __init__(self, config, data_loader, val_data_loader=None):
    self.is_netsc = 'NetSC' in config.inlier_model
    self.requires_e_hat = not self.is_netsc
    if 'PyramidIteration' in config.inlier_model:
      self.requires_e_hat = False
    Trainer.__init__(self, config, data_loader, val_data_loader)
    self.loss = Loss(config, self.device)

  def _initialize_model(self):
    config = self.config

    num_feats = 0
    if 'feats' in config.inlier_feature_type:
      num_feats += config.model_n_out * 2
    elif 'coords' in config.inlier_feature_type:
      num_feats += 4
    elif 'count' in config.inlier_feature_type:
      num_feats += 1
    elif 'ones' == config.inlier_feature_type:
      num_feats = 1

    Model = load_model(config.inlier_model)
    if self.is_netsc:
      model = Model(
          in_channels=num_feats,
          out_channels=1,
          clusters=config.oa_clusters,
          D=4,
      )
    else:
      model = Model(
          num_feats,
          1,
          bn_momentum=config.bn_momentum,
          conv1_kernel_size=config.inlier_conv1_kernel_size,
          normalize_feature=False,
          D=4)
    return model

  def forward(self, input_dict):
    reg_sinput = ME.SparseTensor(
        feats=input_dict['sinput_F'],
        coords=input_dict['sinput_C'],
    ).to(self.device)

    if self.requires_e_hat:
      logit = self.model(reg_sinput).F.squeeze()
      e_hat, _ = compute_e_hat(input_dict['xyz'], logit, input_dict['len_batch'])
      return ([logit], [e_hat])
    else:
      return self.model(reg_sinput, input_dict)

  def _train_epoch(self, epoch):
    gc.collect()

    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.config.iter_size

    loss_meter, prec_meter, recall_meter, f1_meter, ap_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_timer, inlier_timer, total_timer = Timer(), Timer(), Timer()
    
    tot_num_data = len(data_loader_iter) // iter_size
    if self.train_max_iter > 0:
      tot_num_data = min(self.train_max_iter, tot_num_data)
    start_iter = (epoch - 1) * (tot_num_data)

    self.model.train()
    for curr_iter in range(tot_num_data):
      self.optimizer.zero_grad()
      total_timer.tic()

      batch_loss = 0
      for iter_idx in range(iter_size):
        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_timer.toc()

        # Feature extraction
        inlier_timer.tic()
        try:
          logits, e_hats = self.forward(input_dict)
        except RuntimeError:
          print("Runtime error")
          pass
        inlier_timer.toc()

        # Calculate loss
        loss = 0
        for logit, e_hat in zip(logits, e_hats):
          loss_i = self.loss.run(start_iter + curr_iter, input_dict, logit, e_hat)
          loss += loss_i
        loss = loss / iter_size
        loss.backward()

        # Accumulate metrics
        pred = np.hstack(torch.sigmoid(logits[-1].detach()).cpu().numpy())
        target = np.hstack(input_dict['sinput_L'].numpy())
        prec, recall, f1, _ = metrics.precision_recall_fscore_support(
            target, (pred > 0.5).astype(np.int), average='binary')
        ap = metrics.average_precision_score(target, pred)

        prec_meter.update(prec)
        recall_meter.update(recall)
        f1_meter.update(f1)
        ap_meter.update(ap)
        batch_loss += loss.item()
      
      total_timer.toc()
      self.optimizer.step()
      loss_meter.update(batch_loss)
      # Clear
      torch.cuda.empty_cache()

      if curr_iter % self.config.stat_freq == 0:
        # Use the current value to see how stochastic the metrics are
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

        prec_meter.reset()
        recall_meter.reset()
        f1_meter.reset()
        ap_meter.reset()
        loss_meter.reset()
        total_timer.reset()
        data_timer.reset()

  def _valid_epoch(self):
    gc.collect()

    data_loader_iter = self.val_data_loader.__iter__()
    loss_meter, prec_meter, recall_meter, f1_meter, ap_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    ang_errs = []
    ths = np.arange(7) * 5
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
        logits, e_hats = self.forward(input_dict)
        inlier_timer.toc()

        # Calculate loss
        loss = 0
        for i in range(len(logits)):
          loss_i = self.loss.run(curr_iter, input_dict, logits[i], e_hats[i])
          loss += loss_i
        total_timer.toc()

        # Accumulate metrics
        pred = np.hstack(torch.sigmoid(logits[-1].detach()).cpu().numpy())
        target = np.hstack(input_dict['sinput_L'].numpy())
        prec, recall, f1, _ = metrics.precision_recall_fscore_support(
            target, (pred > 0.5).astype(np.int), average='binary')
        ap = metrics.average_precision_score(target, pred)

        prec_meter.update(prec)
        recall_meter.update(recall)
        f1_meter.update(f1)
        ap_meter.update(ap)
        loss_meter.update(loss.item())

        # calcute angular error
        norm_coords = input_dict['norm_coords']
        len_batch = input_dict['len_batch']
        R = input_dict['R'].numpy()
        t = input_dict['t'].numpy()
        e_hat = e_hats[-1].cpu().numpy()
        cursor = 0
        for i, n in enumerate(len_batch):
          _pred = pred[cursor:cursor + n]
          err_q, err_t = compute_angular_error(R[i], t[i], e_hat[i].reshape(3, 3),
                                               norm_coords[i], _pred)
          ang_errs.append(np.maximum(err_q, err_t))

        torch.cuda.empty_cache()

        if curr_iter % self.config.stat_freq == 0:
          hist, _ = np.histogram(ang_errs, ths)
          hist = hist.astype(np.float) / len(ang_errs)
          acc = np.cumsum(hist)
          stat = {
              'prec': prec_meter.avg,
              'recall': recall_meter.avg,
              'f1': f1_meter.avg,
              'ap': ap_meter.avg,
              'mAP5': np.mean(acc[:1]),
              'mAP20': np.mean(acc[:4]),
              'loss': loss_meter.avg
          }
          logging.info(
              ', '.join([f"Validation [{curr_iter}/{tot_num_data}]"] +
                        [f"{k.capitalize()}: {v:.4f}" for k, v in stat.items()] + [
                            f"Data time: {data_timer.avg:.4f}",
                            f"Train time: {total_timer.avg - data_timer.avg:.4f}",
                            f"Total time: {total_timer.avg:.4f}"
                        ]))

    hist, _ = np.histogram(ang_errs, ths)
    hist = hist.astype(np.float) / len(ang_errs)
    acc = np.cumsum(hist)
    stat = {
        'prec': prec_meter.avg,
        'recall': recall_meter.avg,
        'f1': f1_meter.avg,
        'ap': ap_meter.avg,
        'mAP5': np.mean(acc[:1]),
        'mAP20': np.mean(acc[:4]),
        'loss': loss_meter.avg
    }
    logging.info(', '.join([f"Validation"] +
                           [f"{k.capitalize()}: {v:.4f}" for k, v in stat.items()]))

    return stat

  def test(self, test_loader):
    test_iter = test_loader.__iter__()
    logging.info(f"Evaluating on {test_loader.dataset.scene}")

    self.model.eval()
    targets, preds, residuals, err_qs, err_ts = [], [], [], [], []
    with torch.no_grad():
      for _ in range(len(test_iter)):
        input_dict = test_iter.next()

        logits, e_hats = self.forward(input_dict)
        logit = logits[-1].squeeze().cpu()
        e_hat = e_hats[-1].cpu().numpy()

        target = np.hstack(input_dict['sinput_L'].numpy())
        pred = np.hstack(torch.sigmoid(logit))
        norm_coords = np.hstack(input_dict['norm_coords'])
        R = np.hstack(input_dict['R'])
        t = np.hstack(input_dict['t'])

        residual = compute_symmetric_epipolar_residual(
            e_hat.reshape(3, 3).T,
            norm_coords[target.astype(bool), :2],
            norm_coords[target.astype(bool), 2:],
        )

        err_q, err_t = compute_angular_error(R, t, e_hat.reshape(3, 3), norm_coords,
                                             pred)

        targets.append(target)
        preds.append(pred)
        residuals.append(residual)
        err_qs.append(err_q)
        err_ts.append(err_t)
    return targets, preds, residuals, err_qs, err_ts
