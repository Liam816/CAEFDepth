import time
import datetime
import argparse
import sys
import os
import numpy as np
from utils.main_utils import parse_arguments, load_config_file
from model import loader
from data.bts_dataloader import BtsDataLoader
from losses import Depth_Loss
from metrics import AverageMeter, Result
from utils.main_utils import RawAverageMeter
from utils import logger
from data import MyTransforms

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.optim as optim
import multiprocessing

import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

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


def resize(x, resolution):
    tf = transforms.Compose([transforms.Resize(resolution)])
    return tf(x)


eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
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


class Runner():
    def __init__(self, args):
        # NOTE: 训练部分初始化
        print('********** Trainer Initializing **********')
        self.debug = False  # True

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

        if torch.cuda.device_count() > 1:
            gpu_nums = len(cuda_visible.split(','))
            device_ids = [i for i in range(gpu_nums)]
            self.model = nn.DataParallel(model, device_ids=device_ids)
        else:
            self.model = model
        self.model.to(self.device)

        self.resolution_opt = getattr(args, "common.resolution", "full")
        res_dict = resolutions[dataset_name]
        self.resolution = res_dict[self.resolution_opt]
        logger.info("train & evaluation resolution: {}".format(self.resolution))

        data_path = getattr(args, "dataset.root", None)
        eval_mode = getattr(args, "common.eval_mode", "alhashim")
        batch_size = getattr(args, "common.bs", 8)
        # resolution = getattr(args, "common.resolution", "full")
        n_cpus = multiprocessing.cpu_count()
        num_workers = n_cpus // 2
        logger.info("data_path: {}".format(data_path))
        logger.info("batch_size: {}".format(batch_size))
        logger.info("num_workers: {}".format(num_workers))

        # NOTE: 计算模型的参数量
        total = sum([param.nelement() for param in model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

        # NOTE: 测试网络输出形状
        for i in range(1):
            X = torch.randn(size=(8, 3, 480, 640)).to(self.device)
            # X = torch.randn(size=(1, 3, 240, 320)).to(self.device)
            res = model.forward(X)
            print("res.shape:", res.shape)
        exit()

        # NOTE: 测试网络推理速度
        print('********** Testing inference speed **********')
        # _, _ = self.pyTorch_speedtest(self.model, num_test_runs=200)  # 200

        time_list = []
        fps_list = []
        for i in range(3):
            times, fps = self.pyTorch_speedtest(model, num_test_runs=200, seed=i)
            # print('times:{:.6f} fps:{:.2f}'.format(times, fps))
            time_list.append(times)
            fps_list.append(fps)
        # print('average times:{:.6f}s'.format(sum(time_list)/len(time_list)))
        print('average times:{:.3f}ms'.format(1000 * sum(time_list) / len(time_list)))
        print('average fps:{:.2f}'.format(sum(fps_list) / len(fps_list)))
        exit()

        self.model = self.model.train()

        self.resolution_opt = getattr(args, "common.resolution", "full")
        res_dict = resolutions[dataset_name]
        self.resolution = res_dict[self.resolution_opt]
        logger.info("train & evaluation resolution: {}".format(self.resolution))

        print('********** Train & Test Dataloader Initializing **********')
        self.train_loader = BtsDataLoader(args, 'train')
        self.test_loader = BtsDataLoader(args, 'online_eval')

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

        scheduler_step_size = getattr(args, "common.scheduler_step_size", 15)
        logger.info("scheduler_step_size: {}".format(scheduler_step_size))
        # 默认每隔scheduler_step_size个epoch之后，学习率衰减到原来的0.1倍
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                      scheduler_step_size,
                                                      gamma=0.1)

        if eval_mode == 'alhashim':
            self.loss_func = Depth_Loss(0.1, 1, 1, maxDepth=self.maxDepth)
        else:
            self.loss_func = Depth_Loss(1, 0, 0, maxDepth=self.maxDepth)

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
        self.eval_mode = getattr(args, "common.eval_mode", "alhashim")

        checkpoint_path = getattr(args, "common.save_checkpoint", "./checkpoints")
        eval_results_path = os.path.join(checkpoint_path, "eval_results")

        self.result_dir = eval_results_path
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        logger.info("self.eval_mode: {}".format(self.eval_mode))
        logger.info("eval_results_path: {}".format(eval_results_path))

        # 训练的时候根据参数文件中的common.resolution参数来确定训练哪种分辨率的模型
        self.downscale_image = torchvision.transforms.Resize(self.resolution)
        # 模型输出的可能是half分辨率的图片 需要上采样到gt深度图的尺寸才能用以计算loss
        self.upscale_depth = torchvision.transforms.Resize(resolutions[dataset_name]['full'])  # To GT res

        self.to_tensor = MyTransforms.ToTensor(test=True, maxDepth=self.maxDepth)

        self.visualize_images = [0, 100, 200, 300, 400, 500, 600, 605]

        # NOTE: bts参数
        self.do_kb_crop = getattr(args, 'bts.do_kb_crop')
        self.min_depth_eval = getattr(args, 'bts.min_depth_eval')
        self.max_depth_eval = getattr(args, 'bts.max_depth_eval')
        self.garg_crop = getattr(args, 'bts.garg_crop')
        self.eigen_crop = getattr(args, 'bts.eigen_crop')
        self.dataset = getattr(args, 'bts.dataset')

    def train(self, opts):
        torch.cuda.empty_cache()
        self.start_time = time.time()
        for self.epoch in range(self.epoch, self.max_epochs):
            epoch_end = time.time()

            current_time = time.strftime('%H:%M', time.localtime())
            print('{} - Epoch {}'.format(current_time, self.epoch))

            batch_time_meter = RawAverageMeter()
            losses_meter = RawAverageMeter()

            self.train_loop(self.epoch, batch_time_meter, losses_meter)

            if self.epoch > (10 - 1):  # 前10个ckpt都不保存
                self.save_checkpoint()

            logger.log("Train and validate this epoch took {:.2f}min".format((time.time() - epoch_end) / 60.0))

    def eval(self):
        self.model.eval()
        average_meter = AverageMeter()
        print("Evaluating...")
        eval_measures = torch.zeros(10).cuda()
        for _, eval_sample_batched in enumerate(tqdm(self.test_loader.data)):
            with torch.no_grad():
                image = torch.autograd.Variable(eval_sample_batched['image'].cuda(non_blocking=True))
                focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(non_blocking=True))
                gt_depth = eval_sample_batched['depth']
                has_valid_depth = eval_sample_batched['has_valid_depth']
                if not has_valid_depth:
                    # print('Invalid depth. continue.')
                    continue

                # _, _, _, _, pred_depth = model(image, focal)
                pred_depth = self.model(image)

                pred_depth = pred_depth.cpu().numpy().squeeze()
                gt_depth = gt_depth.cpu().numpy().squeeze()

            if self.do_kb_crop:
                height, width = gt_depth.shape
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                pred_depth = pred_depth_uncropped

            pred_depth[pred_depth < self.min_depth_eval] = self.min_depth_eval
            pred_depth[pred_depth > self.max_depth_eval] = self.max_depth_eval
            pred_depth[np.isinf(pred_depth)] = self.max_depth_eval
            pred_depth[np.isnan(pred_depth)] = self.min_depth_eval

            valid_mask = np.logical_and(gt_depth > self.min_depth_eval, gt_depth < self.max_depth_eval)

            if self.garg_crop or self.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if self.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif self.eigen_crop:
                    if self.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)

            measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

            eval_measures[:9] += torch.tensor(measures).cuda()
            eval_measures[9] += 1

    def train_loop(self, curr_epoch, batch_time_meter, losses_meter):
        self.model.train()
        accumulated_loss = 0.0

        batch_nums = len(self.train_loader.data)
        end = time.time()

        for i, data in enumerate(self.train_loader.data):
            image, gt = self.unpack_and_move(data)  # 将样本数据放到gpu上

            self.optimizer.zero_grad()

            prediction = self.model(image)
            # LIAM 如果模型输出非完全尺寸
            prediction = resize(prediction, (image.size(2), image.size(3)))

            loss_value = self.loss_func(prediction, gt)
            loss_value.backward()
            self.optimizer.step()

            accumulated_loss += loss_value.item()

            batch_time_meter.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time_meter.val * (batch_nums - i))))

            if i % 50 == 0:  # 每隔50个batch打印一次
                print('Epoch [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      .format(curr_epoch, i, batch_nums, batch_time=batch_time_meter, eta=eta))
        # Report
        current_time = time.strftime('%H:%M', time.localtime())
        average_loss = accumulated_loss / (len(self.train_loader.data) + 1)
        logger.log('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))
        # print('{} - Average Training Loss: {:3.4f}'.format(current_time, average_loss))

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

    def inverse_depth_norm_test(self, depth):
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
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


def main():
    args = parse_arguments()
    config_file_path = './config/MDE_DDRNet-23-slim_GUB.yaml'
    # config_file_path = './config/MDE_MobileNetV2_LiamEdge.yaml'
    opts = load_config_file(config_file_path, args)

    runner = Runner(opts)
    runner.eval()


if __name__ == '__main__':
    main()



