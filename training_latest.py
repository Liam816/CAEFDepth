import time
import os
import datetime

import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from data import MyTransforms

from data import datasets
from model import loader
from losses import Depth_Loss, silog_loss, Edge_Loss
from metrics import AverageMeter, Result

from utils.main_utils import RawAverageMeter
from utils import logger

from typing import Tuple
import multiprocessing

import shutil
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

from nets.models.MobileOne import reparameterize_model
from nets.models.TopFormer import TokenPyramidModule

from data.bts_dataloader import BtsDataLoader


max_depths = {
    'kitti': 80.0,
    'nyu_reduced': 10.0,
}
nyu_res = {
    'full': (480, 640),
    'half': (240, 320),
    'mini': (224, 224)
}
kitti_res = {
    'full': (384, 1280),
    'half': (192, 640)
}
resolutions = {
    'nyu': nyu_res,
    'nyu_reduced': nyu_res,
    'kitti': kitti_res
}
crops = {
    'kitti': [128, 381, 45, 1196],
    'nyu': [20, 460, 24, 616],
    'nyu_reduced': [20, 460, 24, 616]
}


def resize(x: Tensor, resolution: Tuple):
    tf = transforms.Compose([transforms.Resize(resolution)])
    return tf(x)


def cal_gpu(module):
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    for submodule in module.children():
        if hasattr(submodule, "_parameters"):
            parameters = submodule._parameters
            if "weight" in parameters:
                return parameters["weight"].device
            # if "bias" in parameters:
            #     return parameters["bias"].device


class Trainer():
    def __init__(self, args):
        # torch.manual_seed(816)
        # torch.cuda.manual_seed_all(816)

        # NOTE: 训练部分初始化
        print('********** Trainer Initializing **********')
        self.debug = False  # True

        # self.checkpoint_path = getattr(args, "common.save_checkpoint", "./checkpoints")
        # # self.results_path = self.checkpoint_path + "/best_model"
        # self.results_path = os.path.join(self.checkpoint_path, "best_model")
        # self.config_path = os.path.join(self.checkpoint_path, "config")
        # logger.info("checkpoint_path:{}".format(self.checkpoint_path))
        #
        # if not os.path.isdir(self.checkpoint_path):
        #     os.mkdir(self.checkpoint_path)
        # if not os.path.isdir(self.results_path):
        #     os.mkdir(self.results_path)
        # if not os.path.isdir(self.config_path):
        #     os.mkdir(self.config_path)
        #
        # config_file_path = getattr(args, "--common.config_file_path")
        # shutil.copy(config_file_path, os.path.join(self.config_path, config_file_path.split('/')[-1]))

        self.train_edge = getattr(args, "common.train_edge", False)
        self.use_depthnorm = getattr(args, "common.use_depthnorm", True)

        dataset_name = getattr(args, "dataset.name", "nyu_reduced")
        model_name = getattr(args, "model.name", "GuideDepth")
        logger.info("dataset_name: {}".format(dataset_name))
        logger.info("model_name: {}".format(model_name))

        self.dataset = dataset_name
        self.epoch = 0
        self.val_losses = []
        self.max_epochs = getattr(args, "common.epochs", 20)
        logger.info("Training max epochs: {}".format(self.max_epochs))
        self.maxDepth = max_depths[dataset_name]
        logger.info("Maximum Depth of Dataset: {}".format(self.maxDepth))

        cuda_visible = getattr(args, "common.cuda_visible", '0')
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info("cuda_visible: {}".format(cuda_visible))
        logger.info("self.device: {}".format(self.device))

        model = loader.load_model(args)
        logger.info("Train model: {}".format(model.__class__.__name__))
        # res = cal_gpu(model)
        # print("model.device:{}".format(res))

        if torch.cuda.device_count() > 1:
            gpu_nums = len(cuda_visible.split(','))
            device_ids = [i for i in range(gpu_nums)]
            self.model = nn.DataParallel(model, device_ids=device_ids)
        else:
            self.model = model
        self.model.to(self.device)

        # res = cal_gpu(self.model)
        # print("model.device:{}".format(res))

        self.resolution_opt = getattr(args, "common.resolution", "full")
        res_dict = resolutions[dataset_name]
        self.resolution = res_dict[self.resolution_opt]
        logger.info("train & evaluation resolution: {}".format(self.resolution))

        data_path = getattr(args, "dataset.root", None)
        eval_mode = getattr(args, "common.eval_mode", "alhashim")
        batch_size = getattr(args, "common.bs", 8)
        resolution = getattr(args, "common.resolution", "full")
        n_cpus = multiprocessing.cpu_count()
        num_workers = n_cpus // 2
        logger.info("data_path: {}".format(data_path))
        # logger.info("eval_mode:{}".format(eval_mode))
        logger.info("batch_size: {}".format(batch_size))
        # logger.info("resolution:{}".format(resolution))
        logger.info("num_workers: {}".format(num_workers))

        # NOTE: 计算模型的参数量
        total = sum([param.nelement() for param in model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))
        # exit()

        # 打印网络每层形状和各参数
        # inp_tensor = torch.randint(low=0, high=255, size=(4, 3, 480, 640))  # device=self.device
        # self.model.profile_model(inp_tensor)

        # NOTE: 测试网络输出形状
        # for i in range(1):
        #     X = torch.randn(size=(8, 3, 480, 640)).to(self.device)
        #     # X = torch.randn(size=(1, 3, 240, 320)).to(self.device)
        #     # res = model.forward(X, speed_test=False)
        #     res = model.forward(X)
        #     print("res.shape:", res.shape)
        # exit()

        # for i in range(len(res)):
        #     print('res[{}].shape:{}'.format(i, res[i].shape))
        # exit()
        # for i in range(len(res_list)):
        #     print("layer{}: shape:{}".format(i, res_list[i].size()))

        # TODO: 看一下TopFormer的基于MobileNetV2的backbone推理速度 ！！！
        # model = TokenPyramidModule()

        # NOTE: 测试网络推理速度
        print('********** Testing inference speed **********')
        # _, _ = self.pyTorch_speedtest(self.model, num_test_runs=200)  # 200
        # self.latency_speed_test(model, num_test_runs=200, seed=None)
        # exit()

        # model.eval()
        # model_eval = reparameterize_model(model)

        # time_list = []
        # fps_list = []
        # for i in range(1):
        #     times, fps = self.pyTorch_speedtest(model, num_test_runs=200, seed=i)
        #     # print('times:{:.6f} fps:{:.2f}'.format(times, fps))
        #     time_list.append(times)
        #     fps_list.append(fps)
        # # print('average times:{:.6f}s'.format(sum(time_list)/len(time_list)))
        # print('average times:{:.3f}ms'.format(1000 * sum(time_list) / len(time_list)))
        # print('average fps:{:.2f}'.format(sum(fps_list) / len(fps_list)))
        # # exit()

        self.model = self.model.train()

        print('********** Train & Val Dataloader Initializing **********')
        if dataset_name == 'nyu_reduced':
            self.train_loader = datasets.get_dataloader(dataset_name,
                                                        model_name,
                                                        path=data_path,
                                                        split='train',
                                                        augmentation=eval_mode,
                                                        batch_size=batch_size,
                                                        resolution=resolution,
                                                        workers=num_workers)
            self.val_loader = datasets.get_dataloader(dataset_name,
                                                      model_name,
                                                      path=data_path,
                                                      split='val',
                                                      augmentation=eval_mode,
                                                      batch_size=batch_size,
                                                      resolution=resolution,
                                                      workers=num_workers)
        elif dataset_name == 'kitti':
            self.train_loader = BtsDataLoader(args, 'train')
            self.val_loader = BtsDataLoader(args, 'online_eval')

        learning_rate = getattr(args, "common.lr", 1e-4)
        optimizer = getattr(args, "common.optimizer", "adam")
        logger.info("learning_rate: {}".format(learning_rate))
        logger.info("optimizer: {}".format(optimizer))

        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), learning_rate)
        elif optimizer == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), learning_rate,
                                       momentum=0.9, weight_decay=0.0001)
        else:
            logger.error("optimizer {} is not supported yet.")

        # scheduler_step_size = getattr(args, "common.scheduler_step_size", 15)
        # logger.info("scheduler_step_size: {}".format(scheduler_step_size))
        # # 默认每隔scheduler_step_size个epoch之后，学习率衰减到原来的0.1倍
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
        #                                               scheduler_step_size,
        #                                               gamma=0.1)

        lr_scheduler_type = getattr(args, "common.lr_scheduler", 'cosine')
        logger.info("lr_scheduler: {}".format(lr_scheduler_type))
        self.batch_nums = len(self.train_loader.data)
        if lr_scheduler_type == 'step':
            # 默认每隔scheduler_step_size个epoch之后，学习率衰减到原来的0.1倍
            self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                          step_size=15,
                                                          gamma=0.1)
        elif lr_scheduler_type == 'cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                           T_max=self.max_epochs)
        elif lr_scheduler_type == 'polynomial':
            self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer=self.optimizer,
                                                                      total_iters=self.batch_nums * self.max_epochs)
        elif lr_scheduler_type == 'cyclic':
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer,
                                                                  base_lr=1e-4,
                                                                  max_lr=1e-2,
                                                                  step_size_up=self.batch_nums // 2)
        else:
            print('{} scheduler is not supported yet.'.format(lr_scheduler_type))
            exit()

        self.use_silog = getattr(args, "loss.silog", False)
        if not self.use_silog:
            # if eval_mode == 'alhashim':
            #     self.loss_func = Depth_Loss(0.1, 1, 1, maxDepth=self.maxDepth)
            # else:
            #     self.loss_func = Depth_Loss(1, 0, 0, maxDepth=self.maxDepth)
            self.loss_func = Depth_Loss(1, 0, 0, maxDepth=self.maxDepth)
        else:
            self.loss_func = silog_loss(variance_focus=0.85)  # 参考bts里的默认值

        if self.train_edge:
            self.edge_loss = Edge_Loss(10.0)

        # NOTE: 创建对应的checkpoint文件夹
        self.checkpoint_path = getattr(args, "common.save_checkpoint", "./checkpoints")
        self.results_path = os.path.join(self.checkpoint_path, "best_model")
        self.config_path = os.path.join(self.checkpoint_path, "config")
        logger.info("checkpoint_path: {}".format(self.checkpoint_path))

        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)
        if not os.path.isdir(self.config_path):
            os.mkdir(self.config_path)

        config_file_path = getattr(args, "--common.config_file_path")
        shutil.copy(config_file_path, os.path.join(self.config_path, config_file_path.split('/')[-1]))

        # NOTE: evaluate部分初始化
        print('********** Evaluator Initializing **********')
        self.crop = crops[dataset_name]
        self.eval_mode = getattr(args, "common.eval_mode", "alhashim")

        checkpoint_path = getattr(args, "common.save_checkpoint", "./checkpoints")
        eval_results_path = os.path.join(checkpoint_path, "eval_results")

        self.result_dir = eval_results_path
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        test_path = getattr(args, "dataset.root_test", None)
        self.min_depth_eval = getattr(args, "bts.min_depth_eval", 0.001)
        self.max_depth_eval = getattr(args, "bts.max_depth_eval", 80.0)
        self.do_kb_crop = getattr(args, 'bts.do_kb_crop', True)

        logger.info("self.crop: {}".format(self.crop))
        logger.info("self.eval_mode: {}".format(self.eval_mode))
        logger.info("eval_results_path: {}".format(eval_results_path))
        logger.info("test_dataset_path: {}".format(test_path))
        logger.info("min_depth_eval: {}".format(self.min_depth_eval))
        logger.info("max_depth_eval: {}".format(self.max_depth_eval))

        print('********** Test Dataloader Initializing **********')
        if dataset_name == 'nyu_reduced':
            # test_loader加载的永远都是完全尺寸的图片 在evaluate的时候会先下采样图像再送入网络
            self.test_loader = datasets.get_dataloader(dataset_name,
                                                       model_name,
                                                       path=test_path,
                                                       split='test',
                                                       batch_size=1,
                                                       augmentation=self.eval_mode,
                                                       resolution='full',  # resolution_opt
                                                       workers=num_workers)
        elif dataset_name == 'kitti':
            self.test_loader = BtsDataLoader(args, 'online_eval')

        # 训练的时候根据参数文件中的common.resolution参数来确定训练哪种分辨率的模型
        self.downscale_image = torchvision.transforms.Resize(self.resolution)
        # 模型输出的可能是half分辨率的图片 需要上采样到gt深度图的尺寸才能用以计算loss
        self.upscale_depth = torchvision.transforms.Resize(resolutions[dataset_name]['full'])  # To GT res

        self.to_tensor = MyTransforms.ToTensor(test=True, maxDepth=self.maxDepth)

        self.visualize_images = [0, 100, 200, 300, 400, 500, 600, 605]

    def train(self, opts):
        torch.cuda.empty_cache()
        self.start_time = time.time()

        # 存放训练时验证集指标
        training_metrics_to_save = np.zeros(shape=(8, self.max_epochs))

        for self.epoch in range(self.epoch, self.max_epochs):
            epoch_end = time.time()

            current_time = time.strftime('%H:%M', time.localtime())
            print('{} - Epoch {}'.format(current_time, self.epoch))

            batch_time_meter = RawAverageMeter()
            losses_meter = RawAverageMeter()

            self.train_loop(self.epoch, batch_time_meter, losses_meter)
            # logger.log("pseudo train_loop")
            # exit()

            if self.val_loader is not None:
                results_avg = self.val_loop()

                # precision = 6
                # training_metrics_to_save[0, self.epoch] = np.round(results_avg.rmse, precision)
                # training_metrics_to_save[1, self.epoch] = np.round(results_avg.mae, precision)
                # training_metrics_to_save[2, self.epoch] = np.round(results_avg.delta1, precision)
                # training_metrics_to_save[3, self.epoch] = np.round(results_avg.delta2, precision)
                # training_metrics_to_save[4, self.epoch] = np.round(results_avg.delta3, precision)
                # training_metrics_to_save[5, self.epoch] = np.round(results_avg.absrel, precision)
                # training_metrics_to_save[6, self.epoch] = np.round(results_avg.lg10, precision)
                # training_metrics_to_save[7, self.epoch] = np.round(results_avg.gpu_time, precision)

            if self.epoch > (10 - 1):  # 前10个ckpt都不保存
                self.save_checkpoint()

            logger.log("Train and validate this epoch took {:.2f}min".format((time.time() - epoch_end) / 60.0))

        eval_best_ckpts = False  # False True
        if eval_best_ckpts:
            training_metrics_save_path = os.path.join(self.checkpoint_path, "training_metrics.npy")
            np.save(training_metrics_save_path, training_metrics_to_save)

            """找出最好的checkpoint"""
            rmse = training_metrics_to_save[0, :]
            mae = training_metrics_to_save[1, :]
            delta1 = training_metrics_to_save[2, :]
            delta2 = training_metrics_to_save[3, :]
            delta3 = training_metrics_to_save[4, :]
            absrel = training_metrics_to_save[5, :]
            lg10 = training_metrics_to_save[6, :]
            gpu_time = training_metrics_to_save[7, :]

            res_dict = dict()
            res_dict['rmse'] = rmse
            res_dict['absrel'] = absrel
            res_dict['lg10'] = lg10
            res_dict['delta1'] = delta1
            res_dict['delta2'] = delta2
            res_dict['delta3'] = delta3

            sorted_dict = {}
            for k, v in res_dict.items():
                arr_unique = np.unique(v)
                if 'delta' in k:  # 准确率 需要倒序从大到小排列 最大的排第一 表示效果最好
                    sorted_indices = np.argsort(arr_unique)[::-1] + 1
                else:  # 误差项 默认从小到大排列 最小的排第一 表示效果最好
                    sorted_indices = np.argsort(arr_unique) + 1
                sorted_dict[k] = (arr_unique, sorted_indices)

            ckpt_dict = {}
            for i in range(len(rmse)):  # len(rmse)  有n个ckpt
                temp_list = []
                for k, v in sorted_dict.items():  # 有6个指标
                    for t in range(len(v[0])):  # 有m个候选值
                        if res_dict[k][i] == v[0][t]:
                            temp_list.append(v[1][t])
                ckpt_dict['ckpt_' + str(i)] = (temp_list, temp_list.count(1), sum(temp_list) / 6)

            sorted_ckpt = sorted(ckpt_dict.items(), key=lambda x: x[1][2])
            for i in range(len(sorted_ckpt)):
                print('{}: {}'.format(sorted_ckpt[i][0], sorted_ckpt[i][1]))

            chosen_nums = 4
            chosen_ckpts_list = []
            for i in range(chosen_nums):
                chosen_ckpts_list.append('checkpoint_{}.pth'.format(sorted_ckpt[i][0].split('_')[-1]))
                # chosen_ckpts_list.append('ckpt_{}.pth'.format(sorted_ckpt[i][0].split('_')[-1]))

            print('Start evaluating these chosen best checkpoints...')
            self.evaluate_chosen_ckpts(opts, chosen_ckpts_list)

    def evaluate_chosen_ckpts(self, opts, ckpts_list):
        setattr(opts, 'common.mode', 'eval')

        res_dict = {}
        for ckpt in range(len(ckpts_list)):
            setattr(opts, 'common.best_checkpoint', ckpts_list[ckpt])
            # print('best_ckpt after setattr:{}'.format(getattr(opts, 'common.best_checkpoint')))
            eval_model = loader.load_model(opts)
            eval_model.to(self.device)
            eval_model.eval()
            average_meter = AverageMeter()
            print("Evaluating {}".format(ckpts_list[ckpt]))
            for i, data in enumerate(self.test_loader):
                t0 = time.time()
                image, gt = data
                packed_data = {'image': image[0], 'depth': gt[0]}
                data = self.to_tensor(packed_data)
                image, gt = self.unpack_and_move(data)
                image = image.unsqueeze(0)
                gt = gt.unsqueeze(0)

                image_flip = torch.flip(image, [3])  # [b,c,h,w]将最后一个维度width左右翻转
                gt_flip = torch.flip(gt, [3])
                if self.eval_mode == 'alhashim':
                    # For model input  模型输入的可能是（1/1 1/2 1/4）的原始图片尺寸
                    image = self.downscale_image(image)
                    image_flip = self.downscale_image(image_flip)

                data_time = time.time() - t0
                t0 = time.time()

                inv_prediction = eval_model(image)
                prediction = self.inverse_depth_norm(inv_prediction)

                inv_prediction_flip = self.model(image_flip)
                prediction_flip = self.inverse_depth_norm(inv_prediction_flip)

                gpu_time = time.time() - t0

                if self.eval_mode == 'alhashim':
                    upscale_depth = torchvision.transforms.Resize(gt.shape[-2:])  # To GT res

                    prediction = upscale_depth(prediction)
                    prediction_flip = upscale_depth(prediction_flip)

                    if self.dataset == 'kitti':
                        gt_height, gt_width = gt.shape[-2:]
                        self.crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,
                                              0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

                    # if i in self.visualize_images:
                    #     self.save_image_results(image, gt, prediction, i)

                    gt = gt[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
                    gt_flip = gt_flip[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
                    prediction = prediction[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
                    prediction_flip = prediction_flip[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]

                result = Result()
                result.evaluate(prediction.data, gt.data)
                average_meter.update(result, gpu_time, data_time, image.size(0))

                result_flip = Result()
                result_flip.evaluate(prediction_flip.data, gt_flip.data)
                average_meter.update(result_flip, gpu_time, data_time, image.size(0))

            # Report
            avg = average_meter.average()
            res_dict[ckpts_list[ckpt]] = np.array([avg.delta1, avg.delta2, avg.delta3, avg.rmse, avg.absrel, avg.lg10])

            print('\n*\n'
                  '{} metrics:\n'
                  'RMSE={average.rmse:.4f}\n'
                  'MAE={average.mae:.4f}\n'
                  'Delta1={average.delta1:.4f}\n'
                  'Delta2={average.delta2:.4f}\n'
                  'Delta3={average.delta3:.4f}\n'
                  'REL={average.absrel:.4f}\n'
                  'Lg10={average.lg10:.4f}\n'
                  't_GPU={time:.4f}\n'.format(ckpts_list[ckpt], average=avg, time=avg.gpu_time))

        self.save_results(res_dict)

    # def train_loop(self, curr_epoch, batch_time_meter, losses_meter):
    #     self.model.train()
    #     accumulated_loss = 0.0
    #
    #     if self.dataset == 'nyu_reduced':
    #         batch_nums = len(self.train_loader)
    #         sample_nums = len(self.train_loader.dataset)
    #         loader = self.train_loader
    #     elif self.dataset == 'kitti':
    #         batch_nums = len(self.train_loader.data)
    #         sample_nums = len(self.train_loader.training_samples)
    #         loader = self.train_loader.data
    #
    #     end = time.time()
    #
    #     for i, data in enumerate(loader):
    #         if self.dataset == 'nyu_reduced':
    #             image, depth_gt = self.unpack_and_move(data)  # 将样本数据放到gpu上
    #         elif self.dataset == 'kitti':
    #             image = torch.autograd.Variable(data['image'].cuda(self.device, non_blocking=True))
    #             depth_gt = torch.autograd.Variable(data['depth'].cuda(self.device, non_blocking=True))
    #
    #         self.optimizer.zero_grad()
    #
    #         # 模型输出的是DepthNorm
    #         # 对于NYU数据集 训练集中通过MyTransforms.py里的ToTensor方法将原来的depth转为DepthNorm
    #         inv_prediction = self.model(image)
    #         inv_prediction = resize(inv_prediction, (image.size(2), image.size(3)))  # LIAM 如果模型输出非完全尺寸
    #         prediction = self.inverse_depth_norm(inv_prediction)
    #
    #         inv_prediction_ = inv_prediction.detach().cpu()
    #         has_nan = torch.isnan(inv_prediction_)
    #         nan_count = torch.sum(has_nan).item()
    #         nan_indices = torch.nonzero(has_nan)
    #         nan_indices = nan_indices.numpy()
    #         if has_nan.any():
    #             print('inv_prediction_.shape:{}'.format(inv_prediction_.shape))
    #             print('max:{} min:{}'.format(torch.max(inv_prediction_), torch.min(inv_prediction_)))
    #             print("张量中存在NaN值")
    #             # np.set_printoptions(threshold=np.inf)
    #             # print("NaN值的位置：\n", nan_indices)
    #             print("NaN值的数量：", nan_count)
    #             exit()
    #
    #         e, f = torch.max(inv_prediction), torch.min(inv_prediction)
    #         a, b = torch.max(prediction), torch.min(prediction)
    #         c, d = torch.max(depth_gt), torch.min(depth_gt)
    #
    #         mask = None
    #         if self.dataset == 'nyu_reduced':
    #             mask = depth_gt > 0.1
    #         elif self.dataset == 'kitti':
    #             mask = depth_gt > 1.0
    #
    #         # NOTE: 原版
    #         # if not self.use_silog:
    #         #     loss_value = self.loss_func(prediction, depth_gt, mask.to(torch.bool))
    #         # else:
    #         #     loss_value = self.loss_func.forward(prediction, depth_gt, mask.to(torch.bool))
    #
    #         # NOTE: 参照NYU训练的时候 在train训练集中将gt深度转为depthNorm 网络直接输出的depthNorm
    #         depth_gt_norm = self.depth_norm(depth_gt)
    #         if not self.use_silog:
    #             loss_value = self.loss_func(inv_prediction, depth_gt_norm, mask.to(torch.bool))
    #         else:
    #             loss_value = self.loss_func.forward(inv_prediction, depth_gt_norm, mask.to(torch.bool))
    #
    #         loss_value.backward()
    #
    #         self.optimizer.step()
    #
    #         accumulated_loss += loss_value.item()
    #
    #         batch_time_meter.update(time.time() - end)
    #         end = time.time()
    #         eta = str(datetime.timedelta(seconds=int(batch_time_meter.val * (batch_nums - i))))
    #
    #         # print('iter {}'.format(i+1))
    #         if i % 200 == 0:  # 每隔50个batch打印一次
    #             # print('Epoch [{0}][{1}/{2}]\t'
    #             #       'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
    #             #       'ETA {eta}\t'
    #             #       'LOSS {3:.2f}\t'
    #             #       'pred {4:.2f}/{5:.2f} gt {6:.2f}/{7:.2f}\t'
    #             #       .format(curr_epoch, i, batch_nums, loss_value.item(), a, b, c, d, batch_time=batch_time_meter, eta=eta))
    #
    #             print('Epoch [{0}][{1}/{2}]\t'
    #                   'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
    #                   'ETA {eta}\t'
    #                   'LOSS {3:.2f}\t'
    #                   'inv_pred {4:.4f}/{5:.4f}  pred {6:.4f}/{7:.4f}\t'
    #                   .format(curr_epoch, i, batch_nums, loss_value.item(), e, f, a, b, batch_time=batch_time_meter, eta=eta))
    #
    #     # Report
    #     current_time = time.strftime('%H:%M', time.localtime())
    #     average_loss = accumulated_loss / (sample_nums + 1)
    #     logger.log('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))
    #     # print('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))

    # NOTE: 最近一版
    # def train_loop(self, curr_epoch, batch_time_meter, losses_meter):
    #     self.model.train()
    #     accumulated_loss = 0.0
    #
    #     if self.dataset == 'nyu_reduced':
    #         batch_nums = len(self.train_loader)
    #         sample_nums = len(self.train_loader.dataset)
    #         loader = self.train_loader
    #     elif self.dataset == 'kitti':
    #         batch_nums = len(self.train_loader.data)
    #         sample_nums = len(self.train_loader.training_samples)
    #         loader = self.train_loader.data
    #
    #     end = time.time()
    #     for i, data in enumerate(loader):
    #         if self.train_edge is False:
    #             if self.dataset == 'nyu_reduced':
    #                 image, depth_gt = self.unpack_and_move(data)  # 将样本数据放到gpu上
    #             elif self.dataset == 'kitti':
    #                 image = torch.autograd.Variable(data['image'].cuda(self.device, non_blocking=True))
    #                 depth_gt = torch.autograd.Variable(data['depth'].cuda(self.device, non_blocking=True))
    #
    #             self.optimizer.zero_grad()
    #
    #             # 模型输出的是DepthNorm
    #             # 对于NYU数据集 训练集中通过MyTransforms.py里的ToTensor方法将原来的depth转为DepthNorm
    #             inv_prediction = self.model(image)
    #             inv_prediction = resize(inv_prediction, (image.size(2), image.size(3)))  # LIAM 如果模型输出非完全尺寸
    #             prediction = self.inverse_depth_norm(inv_prediction)
    #
    #             inv_prediction_ = inv_prediction.detach().cpu()
    #             has_nan = torch.isnan(inv_prediction_)
    #             nan_count = torch.sum(has_nan).item()
    #             nan_indices = torch.nonzero(has_nan)
    #             nan_indices = nan_indices.numpy()
    #             if has_nan.any():
    #                 print('inv_prediction_.shape:{}'.format(inv_prediction_.shape))
    #                 print('max:{} min:{}'.format(torch.max(inv_prediction_), torch.min(inv_prediction_)))
    #                 print("张量中存在NaN值")
    #                 print("NaN值的数量：", nan_count)
    #                 exit()
    #
    #             e, f = torch.max(inv_prediction), torch.min(inv_prediction)
    #             a, b = torch.max(prediction), torch.min(prediction)
    #             c, d = torch.max(depth_gt), torch.min(depth_gt)
    #
    #             mask = None
    #             if self.dataset == 'nyu_reduced':
    #                 mask = depth_gt > 0.1
    #             elif self.dataset == 'kitti':
    #                 mask = depth_gt > 1.0
    #
    #             # NOTE: 原版
    #             # if not self.use_silog:
    #             #     loss_value = self.loss_func(prediction, depth_gt, mask.to(torch.bool))
    #             # else:
    #             #     loss_value = self.loss_func.forward(prediction, depth_gt, mask.to(torch.bool))
    #
    #             # NOTE: 参照NYU训练的时候 在train训练集中将gt深度转为depthNorm 网络直接输出的depthNorm
    #             depth_gt_norm = self.depth_norm(depth_gt)
    #             if not self.use_silog:
    #                 loss_value = self.loss_func(inv_prediction, depth_gt_norm, mask.to(torch.bool))
    #             else:
    #                 loss_value = self.loss_func.forward(inv_prediction, depth_gt_norm, mask.to(torch.bool))
    #
    #             loss_value.backward()
    #             self.optimizer.step()
    #             accumulated_loss += loss_value.item()
    #
    #         else:  # NOTE: train edge
    #             if self.dataset == 'nyu_reduced':
    #                 image, depth_gt = self.unpack_and_move(data)  # 将样本数据放到gpu上
    #             elif self.dataset == 'kitti':
    #                 image = torch.autograd.Variable(data['image'].cuda(self.device, non_blocking=True))
    #                 depth_gt = torch.autograd.Variable(data['depth'].cuda(self.device, non_blocking=True))
    #                 gt_edge = torch.autograd.Variable(data['depth_edge'].cuda(non_blocking=True))
    #
    #             self.optimizer.zero_grad()
    #
    #             # 模型输出的是DepthNorm
    #             # 对于NYU数据集 训练集中通过MyTransforms.py里的ToTensor方法将原来的depth转为DepthNorm
    #             inv_prediction, pred_edge = self.model(image)
    #             inv_prediction = resize(inv_prediction, (image.size(2), image.size(3)))  # LIAM 如果模型输出非完全尺寸
    #             prediction = self.inverse_depth_norm(inv_prediction)
    #
    #             inv_prediction_ = inv_prediction.detach().cpu()
    #             has_nan = torch.isnan(inv_prediction_)
    #             nan_count = torch.sum(has_nan).item()
    #             nan_indices = torch.nonzero(has_nan)
    #             nan_indices = nan_indices.numpy()
    #             if has_nan.any():
    #                 print('inv_prediction_.shape:{}'.format(inv_prediction_.shape))
    #                 print('max:{} min:{}'.format(torch.max(inv_prediction_), torch.min(inv_prediction_)))
    #                 print("张量中存在NaN值")
    #                 # np.set_printoptions(threshold=np.inf)
    #                 # print("NaN值的位置：\n", nan_indices)
    #                 print("NaN值的数量：", nan_count)
    #                 exit()
    #
    #             e, f = torch.max(inv_prediction), torch.min(inv_prediction)
    #             a, b = torch.max(prediction), torch.min(prediction)
    #             c, d = torch.max(depth_gt), torch.min(depth_gt)
    #
    #             mask = None
    #             if self.dataset == 'nyu_reduced':
    #                 mask = depth_gt > 0.1
    #             elif self.dataset == 'kitti':
    #                 mask = depth_gt > 1.0
    #
    #             # NOTE: 原版
    #             # if not self.use_silog:
    #             #     loss_value = self.loss_func(prediction, depth_gt, mask.to(torch.bool))
    #             # else:
    #             #     loss_value = self.loss_func.forward(prediction, depth_gt, mask.to(torch.bool))
    #
    #             # NOTE: 参照NYU训练的时候 在train训练集中将gt深度转为depthNorm 网络直接输出的depthNorm
    #             depth_gt_norm = self.depth_norm(depth_gt)
    #             if not self.use_silog:
    #                 loss_value = self.loss_func(inv_prediction, depth_gt_norm, mask.to(torch.bool))
    #                 edge_loss = self.edge_loss(pred_edge, gt_edge)
    #                 loss_value = loss_value + edge_loss
    #             else:
    #                 loss_value = self.loss_func.forward(inv_prediction, depth_gt_norm, mask.to(torch.bool))
    #
    #             loss_value.backward()
    #             self.optimizer.step()
    #             accumulated_loss += loss_value.item()
    #
    #         batch_time_meter.update(time.time() - end)
    #         end = time.time()
    #         eta = str(datetime.timedelta(seconds=int(batch_time_meter.val * (batch_nums - i))))
    #
    #         if i % 200 == 0:  # 每隔50个batch打印一次
    #             print('Epoch [{0}][{1}/{2}]\t'
    #                   'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
    #                   'ETA {eta}\t'
    #                   'LOSS {3:.2f}\t'
    #                   'inv_pred {4:.4f}/{5:.4f}  pred {6:.4f}/{7:.4f}\t'
    #                   .format(curr_epoch, i, batch_nums, loss_value.item(), e, f, a, b, batch_time=batch_time_meter, eta=eta))
    #
    #     # Report
    #     current_time = time.strftime('%H:%M', time.localtime())
    #     average_loss = accumulated_loss / (sample_nums + 1)
    #     logger.log('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))
    #     # print('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))

    def train_loop(self, curr_epoch, batch_time_meter, losses_meter):
        self.model.train()
        accumulated_loss = 0.0

        if self.dataset == 'nyu_reduced':
            batch_nums = len(self.train_loader)
            sample_nums = len(self.train_loader.dataset)
            loader = self.train_loader
        elif self.dataset == 'kitti':
            batch_nums = len(self.train_loader.data)
            sample_nums = len(self.train_loader.training_samples)
            loader = self.train_loader.data

        end = time.time()
        for i, data in enumerate(loader):
            if self.train_edge is False:
                if self.dataset == 'nyu_reduced':
                    image, depth_gt = self.unpack_and_move(data)  # 将样本数据放到gpu上
                elif self.dataset == 'kitti':
                    image = torch.autograd.Variable(data['image'].cuda(self.device, non_blocking=True))
                    depth_gt = torch.autograd.Variable(data['depth'].cuda(self.device, non_blocking=True))

                self.optimizer.zero_grad()

                prediction = self.model(image)
                prediction = resize(prediction, (image.size(2), image.size(3)))  # LIAM 如果模型输出非完全尺寸

                if self.use_depthnorm:
                    # 模型输出的是DepthNorm 需要逆转为直接的深度值
                    prediction = self.inverse_depth_norm(prediction)

                prediction_ = prediction.detach().cpu()
                has_nan = torch.isnan(prediction_)
                nan_count = torch.sum(has_nan).item()
                nan_indices = torch.nonzero(has_nan)
                nan_indices = nan_indices.numpy()
                if has_nan.any():
                    print('prediction_.shape:{}'.format(prediction_.shape))
                    print('max:{} min:{}'.format(torch.max(prediction_), torch.min(prediction_)))
                    print("张量中存在NaN值")
                    print("NaN值的数量：", nan_count)
                    exit()

                a, b = torch.max(prediction), torch.min(prediction)

                mask = None
                if self.dataset == 'nyu_reduced':
                    mask = depth_gt > 0.1
                elif self.dataset == 'kitti':
                    mask = depth_gt > 1.0

                # # NOTE: 参照NYU训练的时候 在train训练集中将gt深度转为depthNorm 网络直接输出的depthNorm
                # depth_gt_norm = self.depth_norm(depth_gt)
                # if not self.use_silog:
                #     loss_value = self.loss_func(prediction, depth_gt_norm, mask.to(torch.bool))
                # else:
                #     loss_value = self.loss_func.forward(prediction, depth_gt_norm, mask.to(torch.bool))

                if not self.use_silog:
                    loss_value = self.loss_func(prediction, depth_gt, mask.to(torch.bool))
                else:
                    loss_value = self.loss_func.forward(prediction, depth_gt, mask.to(torch.bool))

                loss_value.backward()
                self.optimizer.step()
                accumulated_loss += loss_value.item()

            else:  # NOTE: train edge
                if self.dataset == 'nyu_reduced':
                    image, depth_gt = self.unpack_and_move(data)  # 将样本数据放到gpu上
                elif self.dataset == 'kitti':
                    image = torch.autograd.Variable(data['image'].cuda(self.device, non_blocking=True))
                    depth_gt = torch.autograd.Variable(data['depth'].cuda(self.device, non_blocking=True))
                    gt_edge = torch.autograd.Variable(data['depth_edge'].cuda(non_blocking=True))

                self.optimizer.zero_grad()

                # 模型输出的是DepthNorm
                # 对于NYU数据集 训练集中通过MyTransforms.py里的ToTensor方法将原来的depth转为DepthNorm
                inv_prediction, pred_edge = self.model(image)
                inv_prediction = resize(inv_prediction, (image.size(2), image.size(3)))  # LIAM 如果模型输出非完全尺寸
                prediction = self.inverse_depth_norm(inv_prediction)

                inv_prediction_ = inv_prediction.detach().cpu()
                has_nan = torch.isnan(inv_prediction_)
                nan_count = torch.sum(has_nan).item()
                nan_indices = torch.nonzero(has_nan)
                nan_indices = nan_indices.numpy()
                if has_nan.any():
                    print('inv_prediction_.shape:{}'.format(inv_prediction_.shape))
                    print('max:{} min:{}'.format(torch.max(inv_prediction_), torch.min(inv_prediction_)))
                    print("张量中存在NaN值")
                    # np.set_printoptions(threshold=np.inf)
                    # print("NaN值的位置：\n", nan_indices)
                    print("NaN值的数量：", nan_count)
                    exit()

                e, f = torch.max(inv_prediction), torch.min(inv_prediction)
                a, b = torch.max(prediction), torch.min(prediction)
                c, d = torch.max(depth_gt), torch.min(depth_gt)

                mask = None
                if self.dataset == 'nyu_reduced':
                    mask = depth_gt > 0.1
                elif self.dataset == 'kitti':
                    mask = depth_gt > 1.0

                # NOTE: 原版
                # if not self.use_silog:
                #     loss_value = self.loss_func(prediction, depth_gt, mask.to(torch.bool))
                # else:
                #     loss_value = self.loss_func.forward(prediction, depth_gt, mask.to(torch.bool))

                # NOTE: 参照NYU训练的时候 在train训练集中将gt深度转为depthNorm 网络直接输出的depthNorm
                depth_gt_norm = self.depth_norm(depth_gt)
                if not self.use_silog:
                    loss_value = self.loss_func(inv_prediction, depth_gt_norm, mask.to(torch.bool))
                    edge_loss = self.edge_loss(pred_edge, gt_edge)
                    loss_value = loss_value + edge_loss
                else:
                    loss_value = self.loss_func.forward(inv_prediction, depth_gt_norm, mask.to(torch.bool))

                loss_value.backward()
                self.optimizer.step()
                accumulated_loss += loss_value.item()

            batch_time_meter.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time_meter.val * (batch_nums - i))))

            if i % 200 == 0:  # 每隔50个batch打印一次
                print('Epoch [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'LOSS {3:.2f}\t'
                      'pred {4:.4f}/{5:.4f}\t'
                      'lr {6}\t'
                      .format(curr_epoch, i, batch_nums, loss_value.item(), a, b, self.lr_scheduler.get_last_lr()[0],
                              batch_time=batch_time_meter, eta=eta))

        # 每个epoch之后对lr进行迭代
        self.lr_scheduler.step()

        # Report
        current_time = time.strftime('%H:%M', time.localtime())
        average_loss = accumulated_loss / (sample_nums + 1)
        logger.log('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))
        # print('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))

    # def val_loop(self):
    #     torch.cuda.empty_cache()
    #     self.model.eval()
    #     accumulated_loss = 0.0
    #     average_meter = AverageMeter()
    #
    #     if self.dataset == 'nyu_reduced':
    #         loader = self.val_loader
    #         sample_nums = len(self.val_loader.dataset)
    #     elif self.dataset == 'kitti':
    #         loader = self.val_loader.data
    #         sample_nums = len(self.val_loader.testing_samples)
    #
    #     eval_measures = torch.zeros(10).cuda()
    #     with torch.no_grad():
    #         for i, data in enumerate(loader):
    #             # print('i:{} ---------'.format(i))
    #             t0 = time.time()
    #             if self.dataset == 'nyu_reduced':
    #                 image, gt_depth = self.unpack_and_move(data)  # 将样本数据放到gpu上
    #             elif self.dataset == 'kitti':
    #                 image = torch.autograd.Variable(data['image'].cuda(self.device, non_blocking=True))  # [b, 3, 352, 1216]
    #                 # gt_depth = torch.autograd.Variable(data['depth'].cuda(self.device, non_blocking=True))  # [b, 352, 1216, 1]
    #                 gt_depth = data['depth']  # [b, 352, 1216, 1]
    #             data_time = time.time() - t0
    #
    #             # 模型输出的是DepthNorm
    #             pred_inv_depth = self.model(image)
    #             pred_depth = self.inverse_depth_norm(pred_inv_depth)
    #
    #             if self.debug and i == 0:
    #                 self.show_images(image, gt_depth, pred_inv_depth)
    #
    #             # NOTE: 参考bts的online_eval函数
    #             # NOTE: 原版
    #             pred_depth = pred_depth.cpu().numpy().squeeze()  # [1, 1, h, w] -> [h, w]
    #             gt_depth = gt_depth.cpu().numpy().squeeze()  # [1, h, w, 1] -> [h, w]
    #             # NOTE: 参照NYU训练的时候 在train训练集中将gt深度转为depthNorm 网络直接输出的depthNorm
    #             # pred_depth = pred_inv_depth.cpu().numpy().squeeze()  # [1, 1, h, w] -> [h, w]
    #             # gt_depth = self.depth_norm(gt_depth)
    #             # gt_depth = gt_depth.cpu().numpy().squeeze()  # [1, h, w, 1] -> [h, w]
    #
    #             if self.do_kb_crop:
    #                 height, width = gt_depth.shape
    #                 top_margin = int(height - 352)  # [18, 24]
    #                 left_margin = int((width - 1216) / 2)  # [5, 13]
    #                 pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
    #                 pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
    #                 pred_depth = pred_depth_uncropped
    #
    #             pred_depth[pred_depth < self.min_depth_eval] = self.min_depth_eval
    #             pred_depth[pred_depth > self.max_depth_eval] = self.max_depth_eval
    #             pred_depth[np.isinf(pred_depth)] = self.max_depth_eval
    #             pred_depth[np.isnan(pred_depth)] = self.min_depth_eval
    #
    #             valid_mask = np.logical_and(gt_depth > self.min_depth_eval, gt_depth < self.max_depth_eval)
    #
    #             gt_height, gt_width = gt_depth.shape
    #             eval_mask = np.zeros(valid_mask.shape)
    #             eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
    #                       int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1  # eigen crop
    #
    #             valid_mask = np.logical_and(valid_mask, eval_mask)
    #             measures = self.compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
    #
    #             eval_measures[:9] += torch.tensor(measures).cuda()
    #             eval_measures[9] += 1
    #             # NOTE ------------------------------------------------
    #
    #             # NOTE: 原始评估代码
    #             # result = Result()
    #             # result.evaluate(pred_depth.data, gt_depth.data)
    #             # average_meter.update(result, gpu_time, data_time, image.size(0))
    #
    #     # NOTE: 参考bts的online_eval函数
    #     eval_measures_cpu = eval_measures.cpu()
    #     cnt = eval_measures_cpu[9].item()
    #     eval_measures_cpu /= cnt
    #     # print('Computing errors for {} eval samples'.format(int(cnt)))
    #     # print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
    #     #                                                                              'sq_rel', 'log_rms', 'd1', 'd2',
    #     #                                                                              'd3'))
    #     # for i in range(8):
    #     #     print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
    #     # print('{:7.3f}'.format(eval_measures_cpu[8]))
    #
    #     print('Computing errors for {} eval samples'.format(int(cnt)))
    #     metrices_list = ['silog', 'abs_rel', 'log10', 'rmse', 'sq_rel', 'log_rms', 'delta1', 'delta2', 'delta3']
    #     for i in range(9):
    #         print('{:>7}: {:7.3f}'.format(metrices_list[i], eval_measures_cpu[i]), end='\n')
    #
    #     # # NOTE: 原始评估代码
    #     # # Report
    #     # avg = average_meter.average()  # 将每一个batch计算得到的metric结果取平均
    #     # current_time = time.strftime('%H:%M', time.localtime())
    #     # average_loss = accumulated_loss / (sample_nums + 1)
    #     # self.val_losses.append(average_loss)
    #     # print('{} - Average Validation Loss: {:3.4f}'.format(current_time, average_loss))
    #     #
    #     # print('\n*\n'
    #     #       'RMSE={average.rmse:.3f}\n'
    #     #       'MAE={average.mae:.3f}\n'
    #     #       'Delta1={average.delta1:.3f}\n'
    #     #       'Delta2={average.delta2:.3f}\n'
    #     #       'Delta3={average.delta3:.3f}\n'
    #     #       'REL={average.absrel:.3f}\n'
    #     #       'Lg10={average.lg10:.3f}\n'
    #     #       't_GPU={time:.3f}\n'.format(average=avg, time=avg.gpu_time))
    #     #
    #     # return avg
    #
    #     return 1

    def val_loop(self):
        torch.cuda.empty_cache()
        self.model.eval()
        accumulated_loss = 0.0
        average_meter = AverageMeter()

        if self.dataset == 'nyu_reduced':
            loader = self.val_loader
            sample_nums = len(self.val_loader.dataset)
        elif self.dataset == 'kitti':
            loader = self.val_loader.data
            sample_nums = len(self.val_loader.testing_samples)

        eval_measures = torch.zeros(10).cuda()
        with torch.no_grad():
            for i, data in enumerate(loader):
                if self.train_edge is False:
                    t0 = time.time()
                    if self.dataset == 'nyu_reduced':
                        image, gt_depth = self.unpack_and_move(data)  # 将样本数据放到gpu上
                    elif self.dataset == 'kitti':
                        image = torch.autograd.Variable(data['image'].cuda(self.device, non_blocking=True))  # [b, 3, 352, 1216]
                        # gt_depth = torch.autograd.Variable(data['depth'].cuda(self.device, non_blocking=True))  # [b, 352, 1216, 1]
                        gt_depth = data['depth']  # [b, 352, 1216, 1]
                    data_time = time.time() - t0

                    pred_depth = self.model(image)
                    pred_depth_to_show = pred_depth

                    if self.use_depthnorm:
                        # 模型输出的是DepthNorm 需要逆转为直接的深度值
                        pred_depth = self.inverse_depth_norm(pred_depth)

                    if self.debug and i == 0:
                        self.show_images(image, gt_depth, pred_depth_to_show)

                    # NOTE: 参考bts的online_eval函数
                    pred_depth = pred_depth.cpu().numpy().squeeze()  # [1, 1, h, w] -> [h, w]
                    gt_depth = gt_depth.cpu().numpy().squeeze()  # [1, h, w, 1] -> [h, w]

                    if self.do_kb_crop:
                        height, width = gt_depth.shape
                        top_margin = int(height - 352)  # [18, 24]
                        left_margin = int((width - 1216) / 2)  # [5, 13]
                        pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                        pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                        pred_depth = pred_depth_uncropped

                    pred_depth[pred_depth < self.min_depth_eval] = self.min_depth_eval
                    pred_depth[pred_depth > self.max_depth_eval] = self.max_depth_eval
                    pred_depth[np.isinf(pred_depth)] = self.max_depth_eval
                    pred_depth[np.isnan(pred_depth)] = self.min_depth_eval

                    valid_mask = np.logical_and(gt_depth > self.min_depth_eval, gt_depth < self.max_depth_eval)

                    gt_height, gt_width = gt_depth.shape
                    eval_mask = np.zeros(valid_mask.shape)
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                              int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1  # eigen crop

                    valid_mask = np.logical_and(valid_mask, eval_mask)
                    measures = self.compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

                    eval_measures[:9] += torch.tensor(measures).cuda()
                    eval_measures[9] += 1

                else:
                    if self.dataset == 'nyu_reduced':
                        image, gt_depth = self.unpack_and_move(data)  # 将样本数据放到gpu上
                    elif self.dataset == 'kitti':
                        image = torch.autograd.Variable(
                            data['image'].cuda(self.device, non_blocking=True))  # [b, 3, 352, 1216]
                        gt_depth = data['depth']  # [b, 352, 1216, 1]
                        # gt_edge = data['depth_edge']

                    # 模型输出的是DepthNorm
                    pred_inv_depth, _ = self.model(image)
                    pred_depth = self.inverse_depth_norm(pred_inv_depth)

                    if self.debug and i == 0:
                        self.show_images(image, gt_depth, pred_inv_depth)

                    # NOTE: 参考bts的online_eval函数
                    pred_depth = pred_depth.cpu().numpy().squeeze()  # [1, 1, h, w] -> [h, w]
                    gt_depth = gt_depth.cpu().numpy().squeeze()  # [1, h, w, 1] -> [h, w]

                    if self.do_kb_crop:
                        height, width = gt_depth.shape
                        top_margin = int(height - 352)  # [18, 24]
                        left_margin = int((width - 1216) / 2)  # [5, 13]
                        pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                        pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                        pred_depth = pred_depth_uncropped

                    pred_depth[pred_depth < self.min_depth_eval] = self.min_depth_eval
                    pred_depth[pred_depth > self.max_depth_eval] = self.max_depth_eval
                    pred_depth[np.isinf(pred_depth)] = self.max_depth_eval
                    pred_depth[np.isnan(pred_depth)] = self.min_depth_eval

                    valid_mask = np.logical_and(gt_depth > self.min_depth_eval, gt_depth < self.max_depth_eval)

                    gt_height, gt_width = gt_depth.shape
                    eval_mask = np.zeros(valid_mask.shape)
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1  # eigen crop

                    valid_mask = np.logical_and(valid_mask, eval_mask)
                    measures = self.compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

                    eval_measures[:9] += torch.tensor(measures).cuda()
                    eval_measures[9] += 1

        # NOTE: 参考bts的online_eval函数
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt

        print('Computing errors for {} eval samples'.format(int(cnt)))
        metrices_list = ['silog', 'abs_rel', 'log10', 'rmse', 'sq_rel', 'log_rms', 'delta1', 'delta2', 'delta3']
        for i in range(9):
            print('{:>7}: {:7.3f}'.format(metrices_list[i], eval_measures_cpu[i]), end='\n')

        return 1

    def compute_errors(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        d1 = (thresh < 1.25).mean()
        d2 = (thresh < 1.25 ** 2).mean()
        d3 = (thresh < 1.25 ** 3).mean()

        rms = (gt - pred) ** 2
        rms = np.sqrt(rms.mean())

        log_rms = (np.log(gt) - np.log(pred)) ** 2
        log_rms = np.sqrt(log_rms.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        err = np.abs(np.log10(pred) - np.log10(gt))
        log10 = np.mean(err)

        return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.epoch = checkpoint['epoch']

    def save_checkpoint(self):
        # Save checkpoint for training
        checkpoint_dir = os.path.join(self.checkpoint_path,
                                      'checkpoint_{}.pth'.format(self.epoch))
        # checkpoint_dir = os.path.join(self.checkpoint_path,
        #                               'ckpt_{}.pth'.format(self.epoch))

        # torch.save({
        #     'epoch': self.epoch + 1,
        #     'val_losses': self.val_losses,
        #     'model': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     'lr_scheduler': self.lr_scheduler.state_dict(),
        # }, checkpoint_dir)
        torch.save({
            'epoch': self.epoch + 1,
            'val_losses': self.val_losses,
            'model': self.model.state_dict(),
        }, checkpoint_dir)
        current_time = time.strftime('%H:%M', time.localtime())
        print('{} - Model saved'.format(current_time))

    def save_model(self):
        # 默认最后一个模型是最佳的
        best_checkpoint_path = os.path.join(self.checkpoint_path, 'checkpoint_{}.pth'.format(self.max_epochs - 1))
        best_model_pth = os.path.join(self.results_path, 'best_model.pth')

        checkpoint = torch.load(best_checkpoint_path)
        torch.save(checkpoint['model'], best_model_pth)
        print('Model saved.')

    def inverse_depth_norm(self, depth):
        zero_mask = depth == 0.0
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)  # [10/100=0.1, 10]
        depth[zero_mask] = 0.0
        return depth

    def depth_norm(self, depth):
        zero_mask = depth == 0.0
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        depth[zero_mask] = 0.0
        return depth

    def unpack_and_move(self, data):
        if isinstance(data, (tuple, list)):
            image = data[0].to(self.device, non_blocking=True)
            gt = data[1].to(self.device, non_blocking=True)
            return image, gt
        if isinstance(data, dict):  # sample样本是以字典形式保存的
            # keys = data.keys()
            image = data['image'].to(self.device, non_blocking=True)
            gt = data['depth'].to(self.device, non_blocking=True)
            return image, gt
        print('Type not supported')

    def show_images(self, image, gt, pred):
        import matplotlib.pyplot as plt
        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        gt[0, 0, gt[0, 0] == 100.0] = 0.1
        plt.imshow(image_np)
        plt.show()
        plt.imshow(gt[0, 0].cpu())
        plt.show()
        plt.imshow(pred[0, 0].detach().cpu())
        plt.show()

    def pyTorch_speedtest(self, model, num_test_runs=200, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        model.eval()
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        for i in range(num_test_runs + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            x = torch.randn([1, 3, *self.resolution]).cuda()
            torch.cuda.synchronize()  # Synchronize transfer to cuda

            t0 = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times
        # print('[PyTorch] Runtime: {:.6}s'.format(times))
        print('[PyTorch] Runtime: {:.3}ms'.format(times * 1000))
        print('[PyTorch] FPS: {:.2f}'.format(fps))
        return times, fps

    def latency_speed_test(self, model, num_test_runs=200, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        model.eval()
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        times_list = []
        for i in range(num_test_runs + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            x = torch.randn([1, 3, *self.resolution]).cuda()
            torch.cuda.synchronize()  # Synchronize transfer to cuda

            t0 = time.time()
            _, block_times_list = model.forward(x, speed_test=True)
            times_list.append(block_times_list)
            torch.cuda.synchronize()
            times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times
        print('Runtime: {:.6f}s'.format(times))
        print('FPS: {:.2f}'.format(fps))

        times_list = times_list[warm_up_runs:]  # 前几次热身时间算入
        for i in range(len(times_list[0])):  # 3
            temp = 0.0
            for j in range(len(times_list)):  # 200
                temp = temp + times_list[j][i]
            print('Block_{} time: {:.6f}s'.format(i, temp/len(times_list)))

    def save_image_results(self, image, gt, prediction, image_id):
        img = image[0].permute(1, 2, 0).cpu()
        gt = gt[0, 0].permute(0, 1).cpu()
        prediction = prediction[0, 0].permute(0, 1).detach().cpu()
        error_map = gt - prediction
        vmax_error = self.maxDepth / 10.0
        vmin_error = 0.0
        cmap = 'viridis'

        vmax = torch.max(gt[gt != 0.0])
        vmin = torch.min(gt[gt != 0.0])

        save_to_dir = os.path.join(self.result_dir, 'image_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'errors_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        errors = ax.imshow(error_map, vmin=vmin_error, vmax=vmax_error, cmap='Reds')
        fig.colorbar(errors, ax=ax, shrink=0.8)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'gt_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(gt, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'depth_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(prediction, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.savefig(save_to_dir)
        plt.clf()

    def save_results(self, res_dict):
        data_arr = np.zeros(shape=(len(res_dict), 6))  # 有几组checkpoint数据 每组有6个指标
        index_list = []
        columns_list = ['delta1', 'delta2', 'delta3', 'rmse', 'absrel', 'lg10']
        for count, (k, v) in enumerate(res_dict.items()):
            print('k:{} v:{}'.format(k, v))
            index_list.append(k)
            data_arr[count] = v

        data_df = pd.DataFrame(data_arr)
        data_df.columns = columns_list
        data_df.index = index_list
        writer = pd.ExcelWriter(os.path.join(self.checkpoint_path, 'best_ckpts_metrics.xlsx'))  # 创建名称为test的excel表格
        data_df.to_excel(writer, 'page_1', float_format='%.4f')  # float_format精度，将data_df写到test表格的第一页中
        writer.save()  # 保存

    # def save_results(self, average):
    #     results_file = os.path.join(self.result_dir, 'results.txt')
    #     with open(results_file, 'w') as f:
    #         f.write('RMSE,MAE,REL, RMSE_log,Lg10,Delta1,Delta2,Delta3\n')
    #         f.write('{average.rmse:.3f}'
    #                 ',{average.mae:.3f}'
    #                 ',{average.absrel:.3f}'
    #                 ',{average.rmse_log:.3f}'
    #                 ',{average.lg10:.3f}'
    #                 ',{average.delta1:.3f}'
    #                 ',{average.delta2:.3f}'
    #                 ',{average.delta3:.3f}'.format(average=average))
