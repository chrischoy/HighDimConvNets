import os
import os.path as osp
import logging
import numpy as np
import json
from abc import abstractmethod, ABC

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib.util import ensure_dir, count_parameters


class Trainer(ABC):

  def __init__(self, config, data_loader, val_data_loader=None):
    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')
    self.config = config

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare training config
    self.start_epoch = 1
    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.train_max_iter = config.train_max_iter
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq
    self.iter_size = config.iter_size
    self.batch_size = config.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader
    self.test_valid = True if self.val_data_loader is not None else False
    self.pos_weight = config.pos_weight
    self.neg_weight = config.neg_weight

    # Prepare validation config
    self.best_val_comparator = config.best_val_comparator
    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf

    # Initialize model, optimiser and scheduler
    model = self._initialize_model()
    logging.info(model)
    num_params = count_parameters(model)
    logging.info(f"=> Number of parameters = {num_params}")
    self.model = model.to(self.device)
    self.initialize_optimiser_and_scheduler()
    self.resume()

    # Prepare output directory
    self.checkpoint_dir = config.out_dir
    ensure_dir(self.checkpoint_dir)
    json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

    # Intialize tensorboard summary writer
    self.writer = SummaryWriter(logdir=config.out_dir)

  @abstractmethod
  def _initialize_model(self):
    pass

  @abstractmethod
  def _train_epoch(self, epoch):
    pass

  @abstractmethod
  def _valid_epoch(self):
    pass

  def initialize_optimiser_and_scheduler(self):
    config = self.config
    if config.optimizer == 'Adam':
      self.optimizer = getattr(optim, config.optimizer)(
          self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
      self.optimizer = getattr(optim, config.optimizer)(
          self.model.parameters(),
          lr=config.lr,
          momentum=config.momentum,
          weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

  def resume(self):
    config = self.config
    if config.resume is None and config.weights:
      logging.info("=> Loading checkpoint '{}'".format(config.weights))
      checkpoint = torch.load(config.weights)
      if 'state_dict' in checkpoint:
        self.model.load_state_dict(checkpoint['state_dict'])
        logging.info("=> Loaded inlier weights from '{}'".format(config.weights))
      else:
        logging.warn("=> Inlier weight not found in '{}'".format(config.weights))

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info(f"=> Resuming training from the checkpoint '{config.resume}'")
        state = torch.load(config.resume)

        self.start_epoch = state['epoch'] + 1
        logging.info(f"=> Training from {self.start_epoch} to {self.max_epoch}'")
        self.model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])
        logging.info(f"=> Loaded weights, scheduler, optimizer from '{config.resume}'")

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")

  def save_checkpoint(self, epoch, filename='checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)

  def train(self):
    """Full training logic"""

    # Baseline random feature performance
    if self.test_valid:
      val_dict = self._valid_epoch()
      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self.save_checkpoint(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        val_dict = self._valid_epoch()
        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if (self.best_val_comparator == 'larger' and self.best_val < val_dict[self.best_val_metric]) or \
            (self.best_val_comparator == 'smaller' and self.best_val > val_dict[self.best_val_metric]):
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self.save_checkpoint(epoch, 'best_val_checkpoint')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )
