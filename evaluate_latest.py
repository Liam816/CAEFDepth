import time
import os

import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from data import datasets
from model import loader
from metrics import AverageMeter, Result
from data import MyTransforms
from data.bts_dataloader import BtsDataLoader

from utils import logger
import cv2 as cv

max_depths = {
    'kitti': 80.0,
    'nyu': 10.0,
    'nyu_reduced': 10.0,
}
nyu_res = {
    'full': (480, 640),
    'half': (240, 320),
    'mini': (224, 224)}
kitti_res = {
    'full': (384, 1280),
    'tu_small': (128, 416),
    'tu_big': (228, 912),
    'half': (192, 640)}
resolutions = {
    'nyu': nyu_res,
    'nyu_reduced': nyu_res,
    'kitti': kitti_res}
crops = {
    'kitti': [128, 381, 45, 1196],
    'nyu': [20, 460, 24, 616],
    'nyu_reduced': [20, 460, 24, 616]}


class Evaluater():
    def __init__(self, args):
        self.debug = True
        dataset_name = getattr(args, "dataset.name", "nyu_reduced")
        self.dataset = getattr(args, "dataset.name", "nyu_reduced")

        resolution_opt = getattr(args, "common.resolution", "full")
        self.maxDepth = max_depths[self.dataset]
        self.res_dict = resolutions[self.dataset]
        self.resolution = self.res_dict[resolution_opt]
        self.resolution_keyword = resolution_opt
        self.crop = crops[self.dataset]
        self.eval_mode = getattr(args, "common.eval_mode", "alhashim")

        checkpoint_path = getattr(args, "common.save_checkpoint", "./checkpoints")
        eval_results_path = os.path.join(checkpoint_path, "eval_results")

        self.result_dir = eval_results_path
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        logger.info("self.dataset:{}".format(self.dataset))
        logger.info("self.maxDepth:{}".format(self.maxDepth))
        logger.info("self.resolution:{}".format(self.resolution))
        logger.info("self.crop:{}".format(self.crop))
        logger.info("self.eval_mode:{}".format(self.eval_mode))
        logger.info("eval_results_path:{}".format(eval_results_path))

        cuda_visible = getattr(args, "common.cuda_visible", '0')
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info("cuda_visible:{}".format(cuda_visible))
        logger.info("self.device:{}".format(self.device))

        self.model = loader.load_model(args)
        # print("model:", self.model.__class__.__name__)
        self.model.to(self.device)

        # LIAM
        # X = torch.randn(size=(1, 3, 480, 640)).to(self.device)
        # res_list = self.model(X)
        # # for i in range(len(res_list)):
        # #     print("{}_shape:{}".format(i, res_list[i].shape))
        # #     cv.imwrite()
        # #     plt.imsave()
        # exit()

        test_path = getattr(args, "dataset.root_test", None)
        num_workers = getattr(args, "common.num_workers", 10)
        model_name = getattr(args, "model.name", "GuideDepth")
        logger.info("test_path:{}".format(test_path))
        logger.info("num_workers:{}".format(num_workers))
        logger.info("model_name:{}".format(model_name))

        # 计算模型的参数量
        total = sum([param.nelement() for param in self.model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

        if dataset_name == 'nyu_reduced':
            self.test_loader = datasets.get_dataloader(self.dataset,
                                                       model_name,
                                                       path=test_path,
                                                       split='test',
                                                       batch_size=1,
                                                       augmentation=self.eval_mode,
                                                       resolution='full',  # resolution_opt
                                                       workers=num_workers)
        elif dataset_name == 'kitti':
            self.test_loader = BtsDataLoader(args, 'online_eval')
            self.min_depth_eval = getattr(args, "bts.min_depth_eval", 0.001)
            self.max_depth_eval = getattr(args, "bts.max_depth_eval", 80.0)
            self.do_kb_crop = getattr(args, 'bts.do_kb_crop', True)

        self.downscale_image = torchvision.transforms.Resize(self.resolution)  # To Model resolution

        self.to_tensor = MyTransforms.ToTensor(test=True, maxDepth=self.maxDepth)

        # self.visualize_images = [0, 1, 2, 3, 4, 5,
        #                          100, 101, 102, 103, 104, 105,
        #                          200, 201, 202, 203, 204, 205,
        #                          300, 301, 302, 303, 304, 305,
        #                          400, 401, 402, 403, 404, 405,
        #                          500, 501, 502, 503, 504, 505,
        #                          600, 601, 602, 603, 604, 605]
        self.visualize_images = [0, 100, 200, 300, 400, 500, 600, 605]

    def evaluate(self):
        self.model.eval()
        average_meter = AverageMeter()
        print("Evaluating...")

        if self.dataset == 'nyu_reduced':
            loader = self.test_loader
            # sample_nums = len(self.test_loader.dataset)
        elif self.dataset == 'kitti':
            loader = self.test_loader.data
            # sample_nums = len(self.test_loader.testing_samples)

        eval_measures = torch.zeros(10).cuda()
        with torch.no_grad():
            for i, data in enumerate(loader):
                if self.dataset == 'nyu_reduced':
                    t0 = time.time()
                    image, gt = data  # NOTE: nyu_reduced的test_loader直接返回image和depth两个值
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

                    elif self.eval_mode == 'liam':
                        image = self.downscale_image(image)
                        image_flip = self.downscale_image(image_flip)
                        gt = self.downscale_image(gt)
                        gt_flip = self.downscale_image(gt_flip)

                    data_time = time.time() - t0
                    t0 = time.time()

                    inv_prediction = self.model(image)
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

                    elif self.eval_mode == 'liam':

                        if self.dataset == 'kitti':
                            gt_height, gt_width = gt.shape[-2:]
                            self.crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,
                                                  0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

                        # if i in self.visualize_images:
                        #     self.save_image_results(image, gt, prediction, i)

                        # self.crop = [20, 460, 24, 616]
                        self.crop[0] = 10
                        self.crop[1] = 230
                        self.crop[2] = 12
                        self.crop[3] = 308
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

                elif self.dataset == 'kitti':
                    image = data['image'].cuda(self.device, non_blocking=True)  # [b, 3, 352, 1216]
                    # gt_depth = data['depth'].cuda(self.device, non_blocking=True)  # [b, 352, 1216, 1]
                    gt_depth = data['depth']  # [b, 352, 1216, 1]
                    pred_depth = self.model(image)

                    # if i in self.visualize_images:
                    #     self.save_image_results(image, gt_depth.permute(0, 3, 1, 2), pred_depth, i)

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

        # Report
        if self.dataset == 'nyu_reduced':
            avg = average_meter.average()
            self.save_results(avg)
            print('\n*\n'
                  'RMSE={average.rmse:.4f}\n'
                  'MAE={average.mae:.4f}\n'
                  'Delta1={average.delta1:.4f}\n'
                  'Delta2={average.delta2:.4f}\n'
                  'Delta3={average.delta3:.4f}\n'
                  'REL={average.absrel:.4f}\n'
                  'Lg10={average.lg10:.4f}\n'
                  't_GPU={time:.4f}\n'.format(
                average=avg, time=avg.gpu_time))

        elif self.dataset == 'kitti':
            # NOTE: 参考bts的online_eval函数
            eval_measures_cpu = eval_measures.cpu()
            cnt = eval_measures_cpu[9].item()
            eval_measures_cpu /= cnt

            print('Computing errors for {} eval samples'.format(int(cnt)))
            metrices_list = ['silog', 'abs_rel', 'log10', 'rmse', 'sq_rel', 'log_rms', 'delta1', 'delta2', 'delta3']
            for i in range(9):
                print('{:>7}: {:7.3f}'.format(metrices_list[i], eval_measures_cpu[i]), end='\n')

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

    def save_results(self, average):
        results_file = os.path.join(self.result_dir, 'results.txt')
        with open(results_file, 'w') as f:
            f.write('RMSE,MAE,REL, RMSE_log,Lg10,Delta1,Delta2,Delta3\n')
            f.write('{average.rmse:.3f}'
                    ',{average.mae:.3f}'
                    ',{average.absrel:.3f}'
                    ',{average.rmse_log:.3f}'
                    ',{average.lg10:.3f}'
                    ',{average.delta1:.3f}'
                    ',{average.delta2:.3f}'
                    ',{average.delta3:.3f}'.format(
                average=average))

    def inverse_depth_norm(self, depth):
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        return depth

    def depth_norm(self, depth):
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        return depth

    def unpack_and_move(self, data):
        if isinstance(data, (tuple, list)):
            image = data[0].to(self.device, non_blocking=True)
            gt = data[1].to(self.device, non_blocking=True)
            return image, gt
        if isinstance(data, dict):
            keys = data.keys()
            image = data['image'].to(self.device, non_blocking=True)
            gt = data['depth'].to(self.device, non_blocking=True)
            return image, gt
        print('Type not supported')

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
