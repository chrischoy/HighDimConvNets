import lib.twodim_data_loaders
# import lib.threedim_data_loaders


def make_data_loader(config, phase, batch_size, num_workers, shuffle=None, repeat=False):
  if config.dataset in lib.twodim_data_loaders.dataset_str_mapping:
    return lib.twodim_data_loaders.make_data_loader(
        config, phase, batch_size, num_workers, shuffle=shuffle, repeat=repeat)
  # elif config.dataset in lib.threedim_data_loaders.dataset_str_mapping:
  #   return lib.threedim_data_loaders.make_data_loader(
  #       config, phase, batch_size, num_workers, shuffle=shuffle, repeat=repeat)
  else:
    raise ValueError(f'{config.dataset} not defined.')
