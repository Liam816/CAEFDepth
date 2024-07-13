""" 
Depth Loss by Alhashim et al.:

Ibraheem Alhashim, High Quality Monocular Depth Estimation via
Transfer Learning, https://arxiv.org/abs/1812.11941, 2018

https://github.com/ialhashim/DenseDepth
"""

import torch
from torch import nn
import torch.nn.functional as F

from math import exp
import numpy as np


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class Edge_Loss():
    def __init__(self, scale=1.0, crop=None):
        self.loss = torch.nn.L1Loss()
        self.scale = scale
        self.crop = crop

    def __call__(self, pred_edge, gt_edge, edge_indices=None):
        """
        Args:
            pred_edge: 未裁剪的边缘图
            gt_edge: 裁剪后的边缘图
            edge_indices:
        Returns:
        """
        if self.crop is not None:
            pred_edge = pred_edge[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
        assert pred_edge.shape[2] == gt_edge.shape[2] and pred_edge.shape[3] == gt_edge.shape[3], 'sorry!'

        loss = self.loss(pred_edge, gt_edge)
        loss = loss * self.scale

        return loss


class Depth_Loss():
    def __init__(self, alpha, beta, gamma, maxDepth=10.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.maxDepth = maxDepth

        self.L1_Loss = torch.nn.L1Loss()

        print('Depth Loss ------ L1:{} ssim:{} gradient:{}'.format(self.alpha, self.beta, self.gamma))

    def __call__(self, output, depth, mask=None):
        if self.beta == 0 and self.gamma == 0:
            if mask is None:
                valid_mask = depth > 0.0  # 原始操作
                # print('valid_mask.shape:{}'.format(valid_mask.shape))
            else:
                valid_mask = mask
                # print('mask.shape:{}'.format(mask.shape))
            output = output[valid_mask]
            depth = depth[valid_mask]
            l_depth = self.L1_Loss(output, depth)
            # print('l_depth:{}'.format(l_depth))
            # exit()
            loss = l_depth
        else:
            l_depth = self.L1_Loss(output, depth)
            l_ssim = torch.clamp((1 - self.ssim(output, depth, self.maxDepth)) * 0.5, 0, 1)
            l_grad = self.gradient_loss(output, depth)
            # print('l_depth:{}'.format(l_depth))
            # print('l_ssim:{}'.format(l_ssim))
            # print('l_grad:{}'.format(l_grad))
            # exit()
            loss = self.alpha * l_depth + self.beta * l_ssim + self.gamma * l_grad
        return loss

    def ssim(self, img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
        L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)
            padd = window_size // 2

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret

    def gradient_loss(self, gen_frames, gt_frames, alpha=1):
        gen_dx, gen_dy = self.gradient(gen_frames)
        gt_dx, gt_dy = self.gradient(gt_frames)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        # condense into one tensor and avg
        grad_comb = grad_diff_x ** alpha + grad_diff_y ** alpha

        return torch.mean(grad_comb)

    def gradient(self, x):
        """
        idea from tf.image.image_gradients(image)
        https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        """
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top

        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()


class ESC_Loss():
    """
    Edge Salience Score Loss
    """
    def __init__(self, epsilon, crop):
        self.epsilon = epsilon
        self.crop = crop

    def __call__(self, output, depth, indices):
        gt_esc = self.edge_salience_score(depth, indices, crop=self.crop)
        output_esc = self.edge_salience_score(output, indices, crop=self.crop)
        loss = self.epsilon * torch.abs(gt_esc - output_esc)

        return loss

    def edge_salience_score(self, edge_img, indices, crop=None, mode="horizontal"):
        """
        :param edge_img: 网络预测出的单通道的深度图（没有经过裁剪）  [b, c, h, w]
        :param indices: 经过裁剪后的图像中的edge像素点位置 Tuple
        :param crop: 对图像边缘的裁剪，左右上下四个边界，List
        :return:
        """
        if crop is not None:
            edge_img_croped = edge_img[:, :, crop[0]:crop[1], crop[2]:crop[3]]
        else:
            edge_img_croped = edge_img

        _, _, h, w = edge_img_croped.shape  # [b, c, h, w]

        y_loc = indices[0]
        x_loc = indices[1]
        indices_up = (y_loc - 1, x_loc)
        indices_down = (y_loc + 1, x_loc)
        indices_left = (y_loc, x_loc - 1)
        indices_right = (y_loc, x_loc + 1)

        # 防止超出图像边界
        indices_up[0][indices_up[0] > (h - 1)] = h - 1
        indices_up[0][indices_up[0] < 0] = 0
        indices_up[1][indices_up[1] > (w - 1)] = w - 1
        indices_up[1][indices_up[1] < 0] = 0

        indices_down[0][indices_down[0] > (h - 1)] = h - 1
        indices_down[0][indices_down[0] < 0] = 0
        indices_down[1][indices_down[1] > (w - 1)] = w - 1
        indices_down[1][indices_down[1] < 0] = 0

        indices_left[0][indices_left[0] > (h - 1)] = h - 1
        indices_left[0][indices_left[0] < 0] = 0
        indices_left[1][indices_left[1] > (w - 1)] = w - 1
        indices_left[1][indices_left[1] < 0] = 0

        indices_right[0][indices_right[0] > (h - 1)] = h - 1
        indices_right[0][indices_right[0] < 0] = 0
        indices_right[1][indices_right[1] > (w - 1)] = w - 1
        indices_right[1][indices_right[1] < 0] = 0

        assert indices[0].size == indices[1].size, "Sorry!"
        edge_pixel_num = indices[0].size
        # ===================================================

        sum_res_hv = 0
        for i in range(edge_pixel_num):
            center = edge_img_croped[:, :, indices[0][i], indices[1][i]]
            up = edge_img_croped[:, :, indices_up[0][i], indices_up[1][i]]
            down = edge_img_croped[:, :, indices_down[0][i], indices_down[1][i]]
            left = edge_img_croped[:, :, indices_left[0][i], indices_left[1][i]]
            right = edge_img_croped[:, :, indices_right[0][i], indices_right[1][i]]

            temp = torch.abs(center - up) + torch.abs(center - down) + \
                   torch.abs(center - left) + torch.abs(center - right)
            temp = temp / (255 * 4)
            sum_res_hv = sum_res_hv + temp

        # sum_res_h = 0
        # for i in range(edge_pixel_num):
        #     center = edge_img_croped[indices[0][i], indices[1][i], :]
        #     left = edge_img_croped[indices_left[0][i], indices_left[1][i], :]
        #     right = edge_img_croped[indices_right[0][i], indices_right[1][i], :]
        #     temp = np.abs(center - left) + np.abs(center - right)
        #     temp = temp / (255 * 2)
        #     sum_res_h = sum_res_h + temp
        #
        # sum_res_v = 0
        # for i in range(edge_pixel_num):
        #     center = edge_img_croped[indices[0][i], indices[1][i], :]
        #     up = edge_img_croped[indices_up[0][i], indices_up[1][i], :]
        #     down = edge_img_croped[indices_down[0][i], indices_down[1][i], :]
        #     temp = np.abs(center - up) + np.abs(center - down)
        #     temp = temp / (255 * 2)
        #     sum_res_v = sum_res_v + temp


        # sum_res = 0
        # if mode == "horizontal":
        #     for i in range(edge_pixel_num):
        #         center = edge_img_croped[indices[0][i], indices[1][i], :]
        #         left = edge_img_croped[indices_left[0][i], indices_left[1][i], :]
        #         right = edge_img_croped[indices_right[0][i], indices_right[1][i], :]
        #         temp = np.abs(center - left) + np.abs(center - right)
        #         temp = temp / (255 * 2)
        #         sum_res = sum_res + temp
        # elif mode == "vertical":
        #     for i in range(edge_pixel_num):
        #         center = edge_img_croped[indices[0][i], indices[1][i], :]
        #         up = edge_img_croped[indices_up[0][i], indices_up[1][i], :]
        #         down = edge_img_croped[indices_down[0][i], indices_down[1][i], :]
        #         temp = np.abs(center - up) + np.abs(center - down)
        #         temp = temp / (255 * 2)
        #         sum_res = sum_res + temp

        return sum_res_hv  # sum_res_h, sum_res_v



