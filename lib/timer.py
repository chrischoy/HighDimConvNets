import time
import numpy as np


class ConfusionMatrix(object):

  def __init__(self):
    self.eps = np.finfo(float).eps
    self.inlier_meter = AverageMeter()
    self.correspondence_meter = AverageMeter()
    self.reset()

  def update(self, pred, target):
    target = target.astype(np.bool)
    pred_on_pos = pred[target]
    pred_on_neg = pred[~target]

    tp = np.sum(pred_on_pos)
    fn = np.sum(~pred_on_pos)
    fp = np.sum(pred_on_neg)
    tn = np.sum(~pred_on_neg)

    self.tp += tp
    self.fn += fn
    self.fp += fp
    self.tn += tn

    inlier_ratio = np.sum(target) / target.size
    correspondence_accuracy = np.sum(pred) / pred.size

    self.inlier_meter.update(inlier_ratio)
    self.correspondence_meter.update(correspondence_accuracy)

  def eval(self):
    tp, tn, fp, fn, eps = self.tp, self.tn, self.fp, self.fn, self.eps

    accuracy = (tp + tn) / (tp + fp + tn + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    balanced_accuracy = (tpr + tnr) / 2

    return {
        'inlier_ratio': self.inlier_meter.avg,
        'correspondence_accuracy': self.correspondence_meter.avg,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tpr': tpr,
        'tnr': tnr,
        'balanced_accuracy': balanced_accuracy,
    }

  def reset(self):
    self.tp = 0
    self.tn = 0
    self.fp = 0
    self.fn = 0
    self.inlier_meter.reset()
    self.correspondence_meter.reset()


class GroupMeter(object):

  def __init__(self, keys):
    self.keys = keys
    for k in keys:
      setattr(self, k, AverageMeter())

  def update(self, key, value):
    if hasattr(self, key):
      meter = getattr(self, key)
      meter.update(value)
    else:
      raise ValueError(f"{key} is not registered")

  def get(self, key, average=True):
    if hasattr(self, key):
      meter = getattr(self, key)
      if average:
        return meter.avg
      else:
        return meter.val
    else:
      raise ValueError(f"{key} is not registerd")

  def get_dict(self):
    return {k: self.get(k) for k in self.keys}


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0.0
    self.sq_sum = 0.0
    self.count = 0

  def update(self, val, n=1):
    if not np.isnan(val):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count
      self.sq_sum += val**2 * n
      self.var = self.sq_sum / self.count - self.avg**2


class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.avg = 0.

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0
    self.avg = 0

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.avg = self.total_time / self.calls
    if average:
      return self.avg
    else:
      return self.diff
