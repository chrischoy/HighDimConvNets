import warnings

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from util.file import loadh5


def serialize_calibration(path):
  """Load calibration file and serialize

  Args:
    path (str): path to calibration file

  Returns:
    array: serialized 1-d calibration array

  """
  calib_dict = loadh5(path)

  calib_list = []
  calibration_keys = ["K", "R", "T", "imsize"]

  # Flatten calibration data
  for _key in calibration_keys:
    calib_list += [calib_dict[_key].flatten()]

  calib_list += [np.linalg.inv(calib_dict["K"]).flatten()]

  # Serialize calibration data into 1-d array
  calib = np.concatenate(calib_list)
  return calib


def parse_calibration(calib):
  """Parse serialiazed calibration

  Args:
    calib (np.ndarray): serialized calibration

  Returns:
    dict: parsed calibration

  """

  parsed_calib = {}
  parsed_calib["K"] = calib[:9].reshape((3, 3))
  parsed_calib["R"] = calib[9:18].reshape((3, 3))
  parsed_calib["t"] = calib[18:21].reshape(3)
  parsed_calib["imsize"] = calib[21:23].reshape(2)
  parsed_calib["K_inv"] = calib[23:32].reshape((3, 3))
  return parsed_calib


def computeNN(desc0, desc1):
  desc0, desc1 = torch.from_numpy(desc0).cuda(), torch.from_numpy(desc1).cuda()
  d1 = (desc0**2).sum(1)
  d2 = (desc1**2).sum(1)
  distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) -
             2 * torch.matmul(desc0, desc1.transpose(0, 1))).sqrt()
  distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
  nnIdx1 = nnIdx1[:, 0]
  idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.cpu().numpy()]
  return idx_sort


def normalize_keypoint(kp, K, center=None):
  """Normalize keypoint coordinate

  Convert pixel image coordinates into normalized image coordinates

  Args:
    kp (array): list of keypoints
    K (array): intrinsic matrix
    center (array, optional): principal point offset, for LFGC dataset because intrinsic matrix doensn't include principal offset
  Returns:
    array: normalized keypoints as homogenous coordinates

  """
  kp = kp.copy()
  if center is not None:
    kp -= center

  kp = get_homogeneous_coords(kp)
  K_inv = np.linalg.inv(K)
  kp = np.dot(kp, K_inv.T)

  return kp


def build_extrinsic_matrix(R, t):
  """Build extrinsic matrix

  Args:
    R (array): Rotation matrix of shape (3,3)
    t (array): Translation vector of shape (3,)

  Returns:
    array: extrinsic matrix

  """
  return np.vstack((np.hstack((R, t[:, None])), [0, 0, 0, 1]))


def compute_essential_matrix(T0, T1):
  """Compute essential matrix

  Args:
    T0 (array): extrinsic matrix
    T1 (array): extrinsic matrix

  Returns:
    array: essential matrix

  """

  dT = T1 @ np.linalg.inv(T0)
  dR = dT[:3, :3]
  dt = dT[:3, 3]

  skew = skew_symmetric(dt)
  return dR.T @ skew, dR, dt


def skew_symmetric(t):
  """Compute skew symmetric matrix of vector t

  Args:
    t (np.ndarray): vector of shape (3,)

  Returns:
    M (np.ndarray): skew-symmetrix matrix of shape (3, 3)

  """
  M = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
  return M


def get_homogeneous_coords(coords, D=2):
  """Convert coordinates to homogeneous coordinates

  Args:
    coords (array): coordinates
    D (int): dimension. default to 2

  Returns:
    array: homogeneous coordinates

  """

  assert len(coords.shape) == 2, "coords should be 2D array"

  if coords.shape[1] == D + 1:
    return coords
  elif coords.shape[1] == D:
    ones = np.ones((coords.shape[0], 1))
    return np.hstack((coords, ones))
  else:
    raise ValueError("Invalid coordinate dimension")


def compute_symmetric_epipolar_residual(E, coords0, coords1):
  """Compute symmetric epipolar residual

  Symmetric epipolar distance

  Args:
    E (np.ndarray): essential matrix
    coord0 (np.ndarray): homogenous coordinates
    coord1 (np.ndarray): homogenous coordinates

  Returns:
    array: residuals

  """
  with warnings.catch_warnings():
    warnings.simplefilter("error", category=RuntimeWarning)
    coords0 = get_homogeneous_coords(coords0)
    coords1 = get_homogeneous_coords(coords1)

    line_2 = np.dot(E.T, coords0.T)
    line_1 = np.dot(E, coords1.T)

    dd = np.sum(line_2.T * coords1, 1)
    dd = np.abs(dd)

    d = dd * (1.0 / np.sqrt(line_1[0, :]**2 + line_1[1, :]**2 + 1e-7) +
              1.0 / np.sqrt(line_2[0, :]**2 + line_2[1, :]**2 + 1e-7))

    return d


def compute_e_hat(coords, logits, len_batch):
  e_hats = []
  logits_ = []
  residuals = []
  start_idx = 0

  if isinstance(coords, np.ndarray):
    coords = torch.from_numpy(coords).float()

  coords = coords.to(logits.device)
  for npts in len_batch:
    end_idx = start_idx + npts
    coord = coords[start_idx:end_idx]
    logit = logits[start_idx:end_idx]
    e_hat = weighted_8points(
        coord.unsqueeze(0).transpose(2, 1),
        logit.unsqueeze(0),
    )
    e_hats.append(e_hat)
    logits_.append(logit)
    residual = compute_symmetric_epipolar_residual(
        e_hat.reshape(3, 3).detach().cpu(), coord[:, :2].detach().cpu(), coord[:, 2:].detach().cpu())
    residuals.append(torch.from_numpy(residual))
    start_idx = end_idx
  return torch.stack(e_hats, dim=0), torch.cat(residuals, dim=0)


def weighted_8points(coords, logits):
  # logits shape = (batch, num_point, 1)
  # coords shape = (batch, 4, num_point)
  w = torch.nn.functional.relu(torch.tanh(logits))
  X = torch.stack([
      coords[:, 2] * coords[:, 0], coords[:, 2] * coords[:, 1], coords[:, 2],
      coords[:, 3] * coords[:, 0], coords[:, 3] * coords[:, 1], coords[:, 3],
      coords[:, 0], coords[:, 1],
      torch.ones_like(coords[:, 0])
  ],
                  dim=1).transpose(2, 1)
  # wX shape = (batch, num_point, 9)
  wX = w.unsqueeze(-1) * X
  # XwX shape = (batch, 9, 9)
  XwX = torch.bmm(X.transpose(2, 1), wX)

  v = batch_symeig(XwX)
  # _, v = torch.symeig(XwX, eigenvectors=True)
  e = torch.reshape(v[:, :, 0], (logits.shape[0], 9))
  e = e / torch.norm(e, dim=1, keepdim=True)
  return e


def quaternion_from_rotation(R):
  return Rotation.from_matrix(R).as_quat()


def compute_angular_error(R_gt, t_gt, E_hat, coords, scores):
  num_top = len(scores) // 10
  num_top = max(1, num_top)
  th = np.sort(scores)[::-1][num_top]
  mask = scores >= th

  coords = coords.astype(np.float64)
  p1_good = coords[mask, :2]
  p2_good = coords[mask, 2:]
  E_hat = E_hat.astype(p1_good.dtype)

  # decompose estimated essential matrix
  num_inlier, R, t, _ = cv2.recoverPose(E_hat, p1_good, p2_good)

  eps = np.finfo(float).eps

  # calculate rotation error
  q = quaternion_from_rotation(R)
  q = q / (np.linalg.norm(q) + eps)
  q_gt = quaternion_from_rotation(R_gt)
  q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
  loss_q = np.maximum(eps, (1 - np.sum(q * q_gt)**2))
  err_q = np.arccos(1 - 2 * loss_q)

  # calculate translation error
  t = t.flatten()
  t = t / (np.linalg.norm(t) + eps)
  t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
  loss_t = np.maximum(eps, (1 - np.sum(t * t_gt)**2))
  err_t = np.arccos(np.sqrt(1 - loss_t))
  err_q = err_q * 180 / np.pi
  err_t = err_t * 180 / np.pi
  return err_q, err_t


def batch_symeig(X):
  # it is much faster to run symeig on CPU
  X = X.cpu()
  b, d, _ = X.size()
  bv = X.new(b, d, d)
  for batch_idx in range(X.shape[0]):
    e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
    bv[batch_idx, :, :] = v
  bv = bv.cuda()
  return bv


def batch_episym(x1, x2, F):
  batch_size, num_pts = x1.shape[0], x1.shape[1]
  x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts, 1)],
                 dim=-1).reshape(batch_size, num_pts, 3, 1)
  x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts, 1)],
                 dim=-1).reshape(batch_size, num_pts, 3, 1)
  F = F.reshape(-1, 1, 3, 3).repeat(1, num_pts, 1, 1)
  x2Fx1 = torch.matmul(x2.transpose(2, 3),
                       torch.matmul(F, x1)).reshape(batch_size, num_pts)
  Fx1 = torch.matmul(F, x1).reshape(batch_size, num_pts, 3)
  Ftx2 = torch.matmul(F.transpose(2, 3), x2).reshape(batch_size, num_pts, 3)
  ys = x2Fx1**2 * (1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) + 1.0 /
                   (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))
  return ys


def torch_skew_symmetric(v):
  zero = torch.zeros_like(v[:, 0])
  M = torch.stack(
      [zero, -v[:, 2], v[:, 1], v[:, 2], zero, -v[:, 0], -v[:, 1], v[:, 0], zero],
      dim=1)

  return M
