# -*- coding: future_fstrings -*-
import os
import sys
import json
import logging
import torch
from easydict import EasyDict as edict

from lib.trainer import get_trainer
from lib.data_loaders import make_data_loader
from lib.loss import pts_loss2
from config import get_config
from model import load_model

import MinkowskiEngine as ME

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


def main(config, resume=False):
  test_loader = make_data_loader(
      config,
      config.test_phase,
      1,
      num_threads=config.test_num_thread)

  num_feats = 0
  if config.use_color:
    num_feats += 3
  if config.use_normal:
    num_feats += 3
  num_feats = max(1, num_feats)

  Model = load_model(config.model)
  model = Model(num_feats, config.model_n_out, config=config)

  if config.weights:
    logging.info(f"Loading the weights {config.weights}")
    checkpoint = torch.load(config.weights, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

  logging.info(model)

  metrics_fn = [pts_loss2]
  Trainer = get_trainer(config.trainer)
  trainer = Trainer(
      model,
      metrics_fn,
      config=config,
      data_loader=test_loader,
      val_data_loader=test_loader,
  )

  test_dict = trainer._valid_epoch()


if __name__ == "__main__":
  logger = logging.getLogger()
  config = get_config()
  if config.me_num_thread < 0:
    config.me_num_thread = os.cpu_count()

  dconfig = vars(config)
  if config.weights_dir:
    resume_config = json.load(open(config.weights_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['weights_dir', 'dataset'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['weights'] = config.weights_dir + '/checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  ME.initialize_nthreads(config.me_num_thread, D=3)

  main(config)
