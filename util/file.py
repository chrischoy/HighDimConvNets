import re
from os import listdir
from os.path import isfile, isdir, join, splitext

import h5py


def sorted_alphanum(file_list_ordered):
  convert = lambda text: int(text) if text.isdigit() else text
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
  return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
  if extension is None:
    file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
  else:
    file_list = [
        join(path, f)
        for f in listdir(path)
        if isfile(join(path, f)) and splitext(f)[1] == extension
    ]
  file_list = sorted_alphanum(file_list)
  return file_list


def get_file_list_specific(path, string, extension=None):
  if extension is None:
    file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
  elif type(extension) == list:
    file_list = [
        join(path, f)
        for f in listdir(path)
        if isfile(join(path, f)) and string in f and splitext(f)[1] in extension
    ]
    file_list = sorted_alphanum(file_list)
  else:
    file_list = [
        join(path, f)
        for f in listdir(path)
        if isfile(join(path, f)) and string in f and splitext(f)[1] == extension
    ]
    file_list = sorted_alphanum(file_list)
  return file_list


def get_folder_list(path):
  folder_list = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
  folder_list = sorted_alphanum(folder_list)
  return folder_list


def loadh5(path):
  """Load h5 file as dictionary

  Args:
    path (str): h5 file path

  Returns:
    dict_file (dict): loaded dictionary

  """
  try:
    with h5py.File(path, "r") as h5file:
      return readh5(h5file)
  except Exception as e:
    print("Error while loading {}".format(path))
    raise e


def readh5(h5node):
  """Read h5 node recursively and loaded into a dict

  Args:
    h5node (h5py._hl.files.File): h5py File object

  Returns:
    dict_file (dict): loaded dictionary

  """
  dict_file = {}
  for key in h5node.keys():
    if type(h5node[key]) == h5py._hl.group.Group:
      dict_file[key] = readh5(h5node[key])
    else:
      dict_file[key] = h5node[key][...]
  return dict_file


def saveh5(dict_file, target_path):
  """Save dictionary as h5 file

  Args:
    dict_file (dict): dictionary to save
    target_path (str): target path string

  """

  with h5py.File(target_path, "w") as h5file:
    if isinstance(dict_file, list):
      for i, d in enumerate(dict_file):
        newdict = {"dict" + str(i): d}
        writeh5(newdict, h5file)
    else:
      writeh5(dict_file, h5file)


def writeh5(dict_file, h5node):
  """Write dictionaly recursively into h5py file

  Args:
    dict_file (dict): dictionary to write
    h5node (h5py._hl.file.File): target h5py file
  """

  for key in dict_file.keys():
    if isinstance(dict_file[key], dict):
      h5node.create_group(key)
      cur_grp = h5node[key]
      writeh5(dict_file[key], cur_grp)
    else:
      h5node[key] = dict_file[key]
