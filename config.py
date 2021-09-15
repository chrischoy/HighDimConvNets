import argparse
import os.path as osp

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='outputs')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='ContrastiveLossTrainer')
trainer_arg.add_argument('--save_freq_epoch', type=int, default=1)
trainer_arg.add_argument('--batch_size', type=int, default=4)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)
trainer_arg.add_argument('--use_hard_negative', type=str2bool, default=True)
trainer_arg.add_argument('--hard_negative_sample_ratio', type=int, default=0.05)
trainer_arg.add_argument('--hard_negative_max_num', type=int, default=3000)

trainer_arg.add_argument('--num_pos_per_batch', type=int, default=1024)
trainer_arg.add_argument('--num_hn_samples_per_batch', type=int, default=256)

trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--pos_weight', type=float, default=1)
trainer_arg.add_argument('--neg_weight', type=float, default=1)
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")
trainer_arg.add_argument('--stat_freq', type=int, default=40)
trainer_arg.add_argument('--final_test', type=str2bool, default=False)
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--nn_max_n', type=int, default=250)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)
trainer_arg.add_argument('--train_max_iter', type=int, default=2000)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)

trainer_arg.add_argument('--hit_ratio_thresh', type=float, default=0.1)

# Triplets
trainer_arg.add_argument('--triplet_num_pos', type=int, default=256)
trainer_arg.add_argument('--triplet_num_hn', type=int, default=512)
trainer_arg.add_argument('--triplet_num_rand', type=int, default=1024)

# Inlier detection trainer
trainer_arg.add_argument('--inlier_model', type=str, default='ResUNetBN2C')
trainer_arg.add_argument('--inlier_training_start_epoch', type=int, default=-1)
trainer_arg.add_argument('--inlier_feature_type', type=str, default='ones')
trainer_arg.add_argument('--inlier_conv1_kernel_size', type=int, default=3)
trainer_arg.add_argument('--inlier_use_balanced_loss', type=str2bool, default=True)
trainer_arg.add_argument('--registration_min_pairs', type=int, default=100)
trainer_arg.add_argument('--inlier_bin_size', type=int, default=1)
trainer_arg.add_argument('--inlier_logit_thresh', type=float, default=-3)
trainer_arg.add_argument(
    '--inlier_threshold_pixel',
    type=float,
    default=8,
    help='ThreeDMatch inlier threshold in pixel')
trainer_arg.add_argument(
    '--inlier_threshold_type',
    type=str,
    default='hard',
    help='Inlier threshold type: hard, soft')
trainer_arg.add_argument(
    '--inlier_label_type',
    type=str,
    choices=['epipolar', 'projection'],
    default='epipolar',
    help='Inlier label type')
trainer_arg.add_argument(
    '--ucn_inlier_threshold_pixel',
    type=float,
    default=4,
    help='UCN hardest contrastive threshold in pixel')
trainer_arg.add_argument(
    '--ucn_use_sift_kp',
    type=str2bool,
    default=True,
    help='UCN use SIFT keypoints for hardest negative mining')
trainer_arg.add_argument(
    '--threed_feature',
    type=str,
    default='fpfh',
    choices=['fpfh', 'fcgf'],
    help='Features used for training inlier detection')
trainer_arg.add_argument(
  '--ucn_weights',
  type=str,
  help='path to pretrained UCN weights'
)
# Network specific configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='SimpleNetBN2C')
net_arg.add_argument('--model_n_out', type=int, default=32)
net_arg.add_argument('--conv1_kernel_size', type=int, default=3)
net_arg.add_argument('--use_color', type=str2bool, default=False)
net_arg.add_argument('--use_normal', type=str2bool, default=False)
net_arg.add_argument('--normalize_feature', type=str2bool, default=False)
net_arg.add_argument('--dist_type', type=str, default='L2')
net_arg.add_argument('--best_val_metric', type=str, default='feat_match_ratio')
net_arg.add_argument(
    '--best_val_comparator',
    type=str,
    choices=['smaller', 'larger'],
    default='larger',
    help='X the better')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=100)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument(
    '--icp_cache_path', type=str, default="/home/chrischoy/datasets/FCGF/kitti/icp/")

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument(
    '--search_method', type=str, default='gpu', choices=['cpu', 'gpu'])
misc_arg.add_argument(
    '--data_loader_search_method', type=str, default='cpu', choices=['cpu', 'gpu'])
misc_arg.add_argument(
    '--eval_registration',
    type=str2bool,
    default=True,
    help='Evaluate RANSAC registration for a registration network')

misc_arg.add_argument('--weights', type=str, default=None)
misc_arg.add_argument('--weights_dir', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--resume_dir', type=str, default=None)
misc_arg.add_argument('--train_num_workers', type=int, default=2)
misc_arg.add_argument('--val_num_workers', type=int, default=1)
misc_arg.add_argument('--test_num_workers', type=int, default=2)
misc_arg.add_argument('--fast_validation', type=str2bool, default=False)
misc_arg.add_argument(
    '--preselect',
    type=str2bool,
    default=False,
    help='preselect voxelized points to compute normals. Use of voxel_size < 5cm with preselect False is discouraged.'
)

data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ThreeDMatchPairDataset03')
data_arg.add_argument('--voxel_size', type=float, default=0.05)
data_arg.add_argument(
    '--data_dir_25mm',
    type=str,
    default="/home/chrischoy/datasets/FCGF/dataset_full_25")
data_arg.add_argument(
    '--data_dir_10mm', type=str, default="/home/chrischoy/datasets/FCGF/dataset_full")
data_arg.add_argument(
    '--kitti_root', type=str, default="/home/chrischoy/datasets/FCGF/kitti/")
data_arg.add_argument('--use_10mm', type=str2bool, default=False)
data_arg.add_argument(
    '--kitti_max_time_diff',
    type=int,
    default=3,
    help='max time difference between pairs (non inclusive)')
data_arg.add_argument('--kitti_date', type=str, default='2011_09_26')
data_arg.add_argument(
    '--data_dir_3dmatch', type=str, default="/home/chrischoy/datasets/FCGF/3DMatch")
data_arg.add_argument(
    '--collation_3d', type=str, default='collate_pair', help="Collation function type")
# 2D
twod_arg = add_argument_group('2D')
twod_arg.add_argument('--data_dir_2d', type=str, help="path to 2d dataset")
twod_arg.add_argument(
    '--collation_2d', type=str, default='default', help="Collation function type")
twod_arg.add_argument(
    '--obj_num_kp',
    type=int,
    default=2000,
    help="number of keypoints to sample per image")
twod_arg.add_argument(
    '--obj_num_nn', type=int, default=1, help="number of nearest neighbor(s)")
twod_arg.add_argument(
    '--feature_extractor',
    type=str,
    default="sift",
    help="select feature extractor to use")
twod_arg.add_argument(
    '--quantization_size',
    type=float,
    default=0.01,
    help="quantization size to discretize image coordinates")
twod_arg.add_argument(
    '--sample_minimum_coords', type=str2bool, default=False
    )
twod_arg.add_argument(
    '--use_ratio_test',
    type=str2bool,
    default=False,
    help='Use ratio test when matching features')
twod_arg.add_argument(
    '--use_8p', type=str2bool, default=False, help="Use eight-point coordinates")
twod_arg.add_argument('--use_gray', type=str2bool, default=False)
twod_arg.add_argument('--resize_ratio', type=float, default=1.0)
twod_arg.add_argument(
    '--frames_per_one_submap',
    type=int,
    default=200,
    help="Number of frames used to create one fragment")
twod_arg.add_argument(
    '--regression_loss_iter',
    type=int,
    default=20000,
    help="start calculating regression loss after this amount of iteration")
twod_arg.add_argument(
    '--data_dir_raw',
    type=str,
    help="path to raw dataset sources. e.g) the folder that contains ['7-scenes-chess', '7-scenes-fire', ...]"
)
twod_arg.add_argument(
    '--data_dir_processed',
    type=str,
    help="path to preprocessed dataset. e.g) the folder that contains ['7-scenes-chess@seq-01', '7-scenes-fire@seq-01', ...]"
)
twod_arg.add_argument(
    '--pred_threshold',
    type=float,
    default=0.0,
    help="Threshold for inlier prediction confidence")
twod_arg.add_argument(
    '--use_balance_loss',
    type=str2bool,
    default=True,
    help="use balanced classification loss")
twod_arg.add_argument(
    '--post_ransac',
    type=str2bool,
    default=False,
    help="use post ransac on evaluation"
)
# Baseline
baseline_arg = add_argument_group('Baseline')
baseline_arg.add_argument('--baseline_model', type=str, default='Mlesac')
baseline_arg.add_argument('--baseline_num_iter', type=int, default=1000)
baseline_arg.add_argument('--baseline_num_sample', type=int, default=8)
baseline_arg.add_argument('--baseline_inlier_threshold', type=float, default=8)
baseline_arg.add_argument(
    '--baseline_use_normalized_coords', type=str2bool, default=False)
baseline_arg.add_argument('--mlesac_sigma', type=float, default=1.0)
baseline_arg.add_argument('--mlesac_em_iter', type=int, default=10)
baseline_arg.add_argument('--use_parallel', type=str2bool, default=True)
baseline_arg.add_argument(
    '--success_rte_thresh',
    type=float,
    default=0.3,
    help='Success if the RTE below this (m)')
baseline_arg.add_argument(
    '--success_rre_thresh',
    type=float,
    default=15,
    help='Success if the RTE below this (degree)')

# OANet
oanet_arg = add_argument_group('OANet')
oanet_arg.add_argument('--oa_loss_essential', type=float, default=0.1)
oanet_arg.add_argument('--oa_loss_classif', type=float, default=1.0)
oanet_arg.add_argument('--oa_use_fundamental', type=str2bool, default=False)
oanet_arg.add_argument('--oa_obj_geod_th', type=float, default=1e-4)
oanet_arg.add_argument('--oa_geo_loss_margin', type=float, default=0.1)
oanet_arg.add_argument('--oa_loss_essential_init_iter', type=int, default=20000)
oanet_arg.add_argument('--oa_iter_num', type=int, default=1)
oanet_arg.add_argument('--oa_net_depth', type=int, default=12)
oanet_arg.add_argument('--oa_net_channels', type=int, default=128)
oanet_arg.add_argument('--oa_clusters', type=int, default=500)
oanet_arg.add_argument('--oa_use_ratio', type=int, default=0)
oanet_arg.add_argument('--oa_use_mutual', type=int, default=0)

# ND Experiment
nd_arg = add_argument_group('ND')
nd_arg.add_argument('--nd_dataset', type=str, default='HyperLineDataset')
nd_arg.add_argument('--nd_dimension', type=int, default=4)
nd_arg.add_argument('--nd_use_coords_as_feats', type=str2bool, default=True)


def get_config():
  config = parser.parse_args()
  vars(config)['root_dir'] = osp.dirname(osp.abspath(__file__))
  return config


def get_parser():
  return parser
