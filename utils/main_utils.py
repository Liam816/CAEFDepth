import matplotlib
import matplotlib.cm
import numpy as np
import yaml
import os
import argparse
from utils import logger
import collections
from typing import List

try:
    # Workaround for DeprecationWarning when importing Collections
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections


def parse_arguments():
    parser = argparse.ArgumentParser(description='MiT-DenseDepth')

    parser.add_argument('--taskname', type=str, default=None)

    group = parser.add_argument_group(title="common arguments", description="Common arguments")
    group.add_argument("--common.epochs", default=20, type=int, help='number of total epochs to run')
    group.add_argument("--common.lr", "--learning-rate", default=0.0001, type=float, help='initial learning rate')
    group.add_argument("--common.bs", default=8, type=int, help='batch size')
    group.add_argument("--common.cuda_visible", default="0,1", type=str)
    group.add_argument("--common.num_workers", default=None, type=int)
    group.add_argument("--common.enable_coreml_compatible_module", default=False, type=bool)
    group.add_argument("--common.mode", default="train", type=str)
    group.add_argument("--common.save_checkpoint", default="./checkpoints", type=str)
    group.add_argument("--common.best_checkpoint", default="checkpoint_0.pth", type=str)
    group.add_argument('--common.eval_mode', type=str, help='Eval mode', choices=['alhashim', 'tu'], default='alhashim')
    group.add_argument('--common.resolution', type=str, choices=['full', 'half'], default='full')
    group.add_argument('--common.lr_scheduler', type=str, help='scheduler', default='cosine')
    group.add_argument('--common.scheduler_step_size', type=int, help='step size of the scheduler', default=15)
    group.add_argument("--common.optimizer", type=str, default="adam")
    group.add_argument("--common.config_file_path", type=str, default=None)
    group.add_argument("--common.train_edge", type=bool, default=False)
    group.add_argument("--common.use_depthnorm", type=bool, default=True)

    group = parser.add_argument_group(title="dataset arguments", description="dataset arguments")
    group.add_argument("--dataset.name", default=None, type=str)
    group.add_argument("--dataset.root_train", default=None, type=str)
    group.add_argument("--dataset.root_val", default=None, type=str)
    group.add_argument("--dataset.root_test", default=None, type=str)
    group.add_argument("--dataset.root", default=None, type=str)

    group = parser.add_argument_group(title="model arguments", description="model arguments")
    group.add_argument("--model.name", type=str, default="mobilevit", help="Model name")
    group.add_argument("--model.mode", type=str, default="T2", help="Model mode")
    group.add_argument("--model.activation.name", default=None, type=str, help="Non-linear function name (e.g., relu)")
    group.add_argument("--model.activation.inplace", default=False, action="store_true",
                       help="Inplace non-linear functions")
    group.add_argument("--model.activation.neg-slope", default=0.1, type=float, help="Negative slope in leaky relu")
    group.add_argument("--model.mit.mode", type=str, default="small", choices=["xx_small", "x_small", "small"],
                       help="MobileViT mode. Defaults to small")
    group.add_argument("--model.mit.attn-dropout", type=float, default=0.0,
                       help="Dropout in attention layer. Defaults to 0.0")
    group.add_argument("--model.mit.ffn-dropout", type=float, default=0.0,
                       help="Dropout between FFN layers. Defaults to 0.0")
    group.add_argument("--model.mit.dropout", type=float, default=0.0,
                       help="Dropout in Transformer layer. Defaults to 0.0")
    group.add_argument("--model.mit.transformer-norm-layer", type=str, default="layer_norm",
                       help="Normalization layer in transformer. Defaults to LayerNorm")
    group.add_argument("--model.mit.no-fuse-local-global-features", action="store_true",
                       help="Do not combine local and global features in MobileViT block")
    group.add_argument("--model.mit.conv-kernel-size", type=int, default=3,
                       help="Kernel size of Conv layers in MobileViT block")
    group.add_argument("--model.mit.head-dim", type=int, default=None, help="Head dimension in transformer")
    group.add_argument("--model.mit.number-heads", type=int, default=None, help="Number of heads in transformer")
    group.add_argument("--model.gradient-checkpointing", type=bool, default=False)
    group.add_argument("--model.pretrained", type=str, default=None)
    group.add_argument("--model.mobilenetv2.width_multiplier", type=float, default=1.0)

    group = parser.add_argument_group(title="Normalization layers", description="Normalization layers")
    group.add_argument("--model.normalization.name", default=None, type=str,
                       help="Normalization layer. Defaults to None")
    group.add_argument("--model.normalization.momentum", default=0.1, type=float,
                       help="Momentum in normalization layers. Defaults to 0.1")
    group.add_argument("--model.normalization.groups", default=1, type=int)

    group = parser.add_argument_group(title="layers param", description="layers param")
    group.add_argument("--model.layer.global_pool", default="mean", type=str)
    group.add_argument("--model.layer.conv_init", default="kaiming_normal", type=str)
    group.add_argument("--model.layer.linear_init", default="trunc_normal", type=str)
    group.add_argument("--model.layer.linear_init_std_dev", default=0.02, type=float)
    group.add_argument("--model.layer.conv-init", type=str, default="kaiming_normal", help="Init type for conv layers")
    parser.add_argument("--model.layer.conv-init-std-dev", type=float, default=None,
                        help="Std deviation for conv layers")

    group = parser.add_argument_group(title="loss functions", description="loss functions")
    group.add_argument("--loss.name", default=["L1"], type=List)
    group.add_argument("--loss.silog", default=False, type=bool)

    group = parser.add_argument_group(title="bts", description="bts")
    # Dataset
    group.add_argument('--bts.dataset', type=str, help='dataset to train on, kitti or nyu', default='nyu')
    group.add_argument("--bts.use_liam_dataset", default=False, type=bool)
    group.add_argument('--bts.data_path', type=str, help='path to the data')
    group.add_argument('--bts.gt_path', type=str, help='path to the groundtruth data')
    group.add_argument('--bts.filenames_file', type=str, help='path to the filenames text file')
    group.add_argument('--bts.input_height', type=int, help='input height', default=480)
    group.add_argument('--bts.input_width', type=int, help='input width', default=640)
    group.add_argument('--bts.max_depth', type=float, help='maximum depth in estimation', default=10)
    # Preprocessing
    parser.add_argument('--bts.do_random_rotate', help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--bts.degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--bts.do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--bts.use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')
    # Online eval
    parser.add_argument('--bts.do_online_eval', help='if set, perform online eval in every eval_freq steps',
                        action='store_true')
    parser.add_argument('--bts.data_path_eval', type=str, help='path to the data for online evaluation')
    parser.add_argument('--bts.gt_path_eval', type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--bts.filenames_file_eval', type=str, help='path to the filenames text file for online evaluation')
    parser.add_argument('--bts.min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--bts.max_depth_eval', type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--bts.eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--bts.garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--bts.eval_freq', type=int, help='Online evaluation frequency in global steps', default=500)
    parser.add_argument('--bts.eval_summary_directory', type=str, help='output directory for eval summary,'
                                                                   'if empty outputs to checkpoint folder', default='')


    '''原文的命令行参数'''
    #Mode
    # parser.set_defaults(train=False)
    # parser.set_defaults(evaluate=False)
    # parser.add_argument('--train', dest='train', action='store_true')
    # parser.add_argument('--eval', dest='evaluate', action='store_true')

    #Data
    # parser.add_argument('--data_path', type=str, help='path to train data', default="")
    # parser.add_argument('--test_path', type=str, help='path to test data', default="")
    # parser.add_argument('--dataset', type=str, help='dataset for training', choices=['kitti', 'nyu', 'nyu_reduced'],
    #                     default='kitti')
    # parser.add_argument('--resolution', type=str, help='Resolution of the images for training',
    #                     choices=['full', 'half', 'mini', 'tu_small', 'tu_big'],
    #                     default='half')
    # parser.add_argument('--eval_mode', type=str, help='Eval mode', choices=['alhashim', 'tu'], default='alhashim')

    #Model
    # parser.add_argument('--model', type=str, help='name of the model to be trained', default='UpDepth')
    # parser.add_argument('--weights_path', type=str, help='path to model weights')

    #Checkpoint
    # parser.add_argument('--load_checkpoint', type=str, help='path to checkpoint', default='')
    # parser.add_argument('--save_checkpoint', type=str, help='path to save checkpoints to', default='./checkpoints')
    # parser.add_argument('--save_results', type=str, help='path to save results to', default='./results')

    #Optimization
    # parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    # parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-4)
    # parser.add_argument('--num_epochs', type=int, help='number of epochs', default=20)
    # parser.add_argument('--scheduler_step_size', type=int, help='step size of the scheduler', default=15)

    #System
    # parser.add_argument('--num_workers', type=int, help='number of dataloader workers', default=2)
    # parser.add_argument("--cuda-visible", type=str, default="0,1")

    opts = parser.parse_args()

    return opts


def flatten_yaml_as_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections_abc.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_file(config_file_path: str, opts):
    with open(config_file_path, "r") as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
            flat_cfg = flatten_yaml_as_dict(cfg)
            # logger.info("LIAM flat_cfg:{}".format(flat_cfg))
            for k, v in flat_cfg.items():
                if hasattr(opts, k):
                    # print("k:", k, "v:", v)
                    setattr(opts, k, v)
            setattr(opts, "--common.config_file_path", config_file_path)
        except yaml.YAMLError as exc:
            logger.error(
                "Error while loading config file: {}. Error message: {}".format(
                    config_file_path, str(exc)
                )
            )
    return opts


def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth


class RawAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))
