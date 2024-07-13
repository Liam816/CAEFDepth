import time
import os
import datetime

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms

from data import datasets
from model import loader
from losses import Depth_Loss
from metrics import AverageMeter, Result

from utils.main_utils import RawAverageMeter
from utils import logger

from typing import Tuple
import multiprocessing

import shutil
import cv2 as cv
import matplotlib.pyplot as plt

max_depths = {
    'kitti': 80.0,
    'nyu_reduced': 10.0,
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
        self.debug = False  # True

        self.checkpoint_pth = getattr(args, "common.save_checkpoint", "./checkpoints")
        # self.results_pth = self.checkpoint_pth + "/best_model"
        self.results_pth = os.path.join(self.checkpoint_pth, "best_model")
        self.config_pth = os.path.join(self.checkpoint_pth, "config")
        logger.info("checkpoint_pth:{}".format(self.checkpoint_pth))

        if not os.path.isdir(self.checkpoint_pth):
            os.mkdir(self.checkpoint_pth)
        if not os.path.isdir(self.results_pth):
            os.mkdir(self.results_pth)
        if not os.path.isdir(self.config_pth):
            os.mkdir(self.config_pth)

        config_file_path = getattr(args, "--common.config_file_path")
        shutil.copy(config_file_path, os.path.join(self.config_pth, config_file_path.split('/')[-1]))

        dataset_name = getattr(args, "dataset.name", "nyu_reduced")
        model_name = getattr(args, "model.name", "GuideDepth")
        logger.info("dataset_name:{}".format(dataset_name))
        logger.info("model_name:{}".format(model_name))

        self.epoch = 0
        self.val_losses = []
        self.max_epochs = getattr(args, "common.epochs", 20)
        logger.info("Training max epochs:{}".format(self.max_epochs))
        self.maxDepth = max_depths[dataset_name]
        logger.info("Maximum Depth of Dataset: {}".format(self.maxDepth))

        cuda_visible = getattr(args, "common.cuda_visible", '0')
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info("cuda_visible:{}".format(cuda_visible))
        logger.info("self.device:{}".format(self.device))

        model = loader.load_model(args)
        logger.info("Train model:{}".format(model.__class__.__name__))
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

        # 计算模型的参数量
        total = sum([param.nelement() for param in model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

        # temp_net = model.rgb_feature_extractor
        # for name, param in temp_net.named_parameters():
        #     print(name, param.shape)

        # exit()

        data_path = getattr(args, "dataset.root", None)
        eval_mode = getattr(args, "common.eval_mode", "alhashim")
        batch_size = getattr(args, "common.bs", 8)
        resolution = getattr(args, "common.resolution", "full")
        n_cpus = multiprocessing.cpu_count()
        num_workers = n_cpus // 4
        logger.info("data_path:{}".format(data_path))
        logger.info("eval_mode:{}".format(eval_mode))
        logger.info("batch_size:{}".format(batch_size))
        logger.info("resolution:{}".format(resolution))
        logger.info("num_workers:{}".format(num_workers))

        # # 打印网络每层形状和各参数
        # inp_tensor = torch.randint(low=0, high=255, size=(4, 3, 480, 640))  # device=self.device
        # self.model.profile_model(inp_tensor)

        # 测试网络输出形状
        model.cuda()
        X = torch.randn(size=(4, 3, 480, 640)).cuda()
        res = model.forward(X)
        print("res.shape:", res.shape)
        # for i in range(len(res_list)):
        #     print("layer{}: shape:{}".format(i, res_list[i].size()))

        exit()

        self.train_loader = datasets.get_dataloader(dataset_name,
                                                    model_name,
                                                    path=data_path,
                                                    split='train',
                                                    augmentation=eval_mode,
                                                    batch_size=batch_size,
                                                    resolution=resolution,
                                                    workers=num_workers)

        # data = next(iter(self.train_loader))
        # edge_gt = data["edge_gt"].numpy()
        # print("edge_gt.shape:", edge_gt.shape)
        # print("edge_gt[:1, :, :].shape:", edge_gt[:1, :, :].shape)
        # print("type(edge_gt)", type(edge_gt))
        # # np.set_printoptions(threshold=np.inf)
        # # print(edge_gt[:1, :, :])
        # plt.imshow(edge_gt[:1, :, :].transpose(1, 2, 0), cmap="Greys")
        # plt.show()
        # exit()

        # images = data["image"]*255
        # images = images.numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        # print("images.shape:", images.shape)
        #
        # plt.subplot(1, 2, 1)
        # plt.imshow(images[0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(images[1])
        # plt.show()
        #
        # exit()

        # self.val_loader = datasets.get_dataloader(dataset_name,
        #                                           model_name,
        #                                           path=data_path,
        #                                           split='val',
        #                                           augmentation=eval_mode,
        #                                           batch_size=batch_size,
        #                                           resolution=resolution,
        #                                           workers=num_workers)

        learning_rate = getattr(args, "common.lr", 1e-4)
        optimizer = getattr(args, "common.optimizer", "adam")
        logger.info("learning_rate:{}".format(learning_rate))
        logger.info("optimizer:{}".format(optimizer))

        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), learning_rate)
        elif optimizer == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), learning_rate,
                                       momentum=0.9, weight_decay=0.0001)
        else:
            logger.error("optimizer {} is not supported yet.")

        scheduler_step_size = getattr(args, "common.scheduler_step_size", 15)
        logger.info("scheduler_step_size:{}".format(scheduler_step_size))
        # 默认每隔scheduler_step_size个epoch之后，学习率衰减到原来的0.1倍
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                      scheduler_step_size,
                                                      gamma=0.1)

        if eval_mode == 'alhashim':
            self.loss_func = Depth_Loss(0.1, 1, 1, maxDepth=self.maxDepth)
        else:
            self.loss_func = Depth_Loss(1, 0, 0, maxDepth=self.maxDepth)

        # # Load Checkpoint
        # if args.load_checkpoint != '':
        #     self.load_checkpoint(args.load_checkpoint)

        # # LIAM
        # with torch.no_grad():
        #     for i, data in enumerate(self.val_loader):
        #         t0 = time.time()
        #         image, gt = self.unpack_and_move(data)  # 将样本数据放到gpu上
        #         data_time = time.time() - t0
        #
        #         # LIAM
        #         print("gt.max:{}".format(torch.max(gt)))
        #         print("gt.min:{}".format(torch.min(gt)))
        #         exit()


    def train(self):
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

            # logger.log("pseudo train_loop")
            self.train_loop(self.epoch, batch_time_meter, losses_meter)

            # if self.val_loader is not None:
            #     # logger.log("pseudo val_loop")
            #     results_avg = self.val_loop()
            #
            #     training_metrics_to_save[0, self.epoch] = results_avg.rmse
            #     training_metrics_to_save[1, self.epoch] = results_avg.mae
            #     training_metrics_to_save[2, self.epoch] = results_avg.delta1
            #     training_metrics_to_save[3, self.epoch] = results_avg.delta2
            #     training_metrics_to_save[4, self.epoch] = results_avg.delta3
            #     training_metrics_to_save[5, self.epoch] = results_avg.absrel
            #     training_metrics_to_save[6, self.epoch] = results_avg.lg10
            #     training_metrics_to_save[7, self.epoch] = results_avg.gpu_time

            # self.save_checkpoint()

            logger.log("Train and validate this epoch took {:.2f}min".format( (time.time()-epoch_end)/60.0 ))
            # print("Train&val this epoch took {:.2f}min".format( (time.time()-epoch_end)/60.0 ))

        self.save_model()
        training_metrics_save_path = os.path.join(self.checkpoint_pth, "training_metrics.npy")
        np.save(training_metrics_save_path, training_metrics_to_save)

    def train_loop(self, curr_epoch, batch_time_meter, losses_meter):
        self.model.train()
        accumulated_loss = 0.0

        batch_nums = len(self.train_loader)
        end = time.time()

        for i, data in enumerate(self.train_loader):
            image, gt = self.unpack_and_move(data)  # 将样本数据放到gpu上
            self.optimizer.zero_grad()

            prediction = self.model(image)
            exit()
        #     # LIAM 如果模型输出非完全尺寸
        #     prediction = resize(prediction, (image.size(2), image.size(3)))
        #
        #     loss_value = self.loss_func(prediction, gt)
        #     loss_value.backward()
        #     self.optimizer.step()
        #
        #     accumulated_loss += loss_value.item()
        #
        #     batch_time_meter.update(time.time() - end)
        #     end = time.time()
        #     eta = str(datetime.timedelta(seconds=int(batch_time_meter.val * (batch_nums - i))))
        #
        #     if i % 50 == 0:  # 每隔50个batch打印一次
        #         print('Epoch [{0}][{1}/{2}]\t'
        #               'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
        #               'ETA {eta}\t'
        #               .format(curr_epoch, i, batch_nums, batch_time=batch_time_meter, eta=eta))
        # # Report
        # current_time = time.strftime('%H:%M', time.localtime())
        # average_loss = accumulated_loss / (len(self.train_loader.dataset) + 1)
        # logger.log('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))
        # # print('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))

    def val_loop(self):
        torch.cuda.empty_cache()
        self.model.eval()
        accumulated_loss = 0.0
        average_meter = AverageMeter()

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                t0 = time.time()
                image, gt = self.unpack_and_move(data)  # 将样本数据放到gpu上
                data_time = time.time() - t0

                t0 = time.time()
                inv_prediction = self.model(image)
                # LIAM 如果模型输出非完全尺寸
                inv_prediction = resize(inv_prediction, (image.size(2), image.size(3)))

                prediction = self.inverse_depth_norm(inv_prediction)
                gpu_time = time.time() - t0

                if self.debug and i == 0:
                    self.show_images(image, gt, prediction)

                loss_value = self.loss_func(inv_prediction, self.depth_norm(gt))
                accumulated_loss += loss_value.item()

                result = Result()
                result.evaluate(prediction.data, gt.data)
                average_meter.update(result, gpu_time, data_time, image.size(0))

        # Report
        avg = average_meter.average()  # 将每一个batch计算得到的metric结果取平均
        current_time = time.strftime('%H:%M', time.localtime())
        average_loss = accumulated_loss / (len(self.val_loader.dataset) + 1)
        self.val_losses.append(average_loss)
        print('{} - Average Validation Loss: {:3.4f}'.format(current_time, average_loss))

        print('\n*\n'
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'Delta1={average.delta1:.3f}\n'
              'Delta2={average.delta2:.3f}\n'
              'Delta3={average.delta3:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'Lg10={average.lg10:.3f}\n'
              't_GPU={time:.3f}\n'.format(
            average=avg, time=avg.gpu_time))

        return avg

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.epoch = checkpoint['epoch']

    def save_checkpoint(self):
        # Save checkpoint for training
        checkpoint_dir = os.path.join(self.checkpoint_pth,
                                      'checkpoint_{}.pth'.format(self.epoch))
        torch.save({
            'epoch': self.epoch + 1,
            'val_losses': self.val_losses,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }, checkpoint_dir)
        current_time = time.strftime('%H:%M', time.localtime())
        print('{} - Model saved'.format(current_time))

    def save_model(self):
        # 默认最后一个模型是最佳的
        best_checkpoint_pth = os.path.join(self.checkpoint_pth, 'checkpoint_{}.pth'.format(self.max_epochs-1))
        best_model_pth = os.path.join(self.results_pth, 'best_model.pth')

        checkpoint = torch.load(best_checkpoint_pth)
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
