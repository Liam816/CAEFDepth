import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.DDRNet_23_slim import DualResNet_Backbone
from nets.modules import BaseUpsamplingBlock
from model.modules import Guided_Upsampling_Block
from ..modules.cbam import CBAM
from ..modules.liam_upsample_block import FasterMutableBlock

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time


def normalize_np(x: np.ndarray):
    min_val = np.min(x)
    max_val = np.max(x)
    res = (x - min_val) / (max_val - min_val)
    return res.astype(np.float32)


def normalize2img_tensor(x: Tensor):
    min_val = x.min()
    max_val = x.max()
    res = (x - min_val) / (max_val - min_val)
    res = res * 255.
    return res


def edge_extractor(x: Tensor, mode, device='cuda:0'):
    b, c, h, w = x.size()
    x_ = x
    x_ = x_ * 255
    x_ = x_.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)  # [b, h, w, c]

    if mode == 'sobel':
        # NOTE: Sobel
        edge_batch_tensor = torch.randn(size=(b, 3, h, w))
        for i in range(b):
            Sobelx = cv.Sobel(x_[i, :, :, :], cv.CV_8U, 1, 0)  # 输出unit8类型的图像
            Sobely = cv.Sobel(x_[i, :, :, :], cv.CV_8U, 0, 1)
            Sobelx = cv.convertScaleAbs(Sobelx)
            Sobely = cv.convertScaleAbs(Sobely)
            Sobelxy = cv.addWeighted(Sobelx, 0.5, Sobely, 0.5, 0)  # [h, w, 3]
            Sobelxy = Sobelxy.transpose(2, 0, 1)  # [3, h, w]
            edge_batch_tensor[i, :, :, :] = torch.from_numpy(Sobelxy).type(torch.float32)
        edge = edge_batch_tensor.to(device)  # [b, 3, h, w]
    elif mode == 'canny':
        # NOTE: Canny
        edge_batch_tensor = torch.randn(size=(b, 1, h, w))
        for i in range(b):
            canny_edge = cv.Canny(x[i, :, :, :], 100, 200)
            canny_edge = np.expand_dims(canny_edge, axis=0)  # [1, h, w]
            canny_edge = np.expand_dims(canny_edge, axis=0)  # [1, 1, h, w]
            canny_edge = normalize_np(canny_edge)  # 将数据缩放到[0, 1]区间
            edge_batch_tensor[i, :, :, :] = torch.from_numpy(canny_edge).type(torch.float32)
        edge = edge_batch_tensor.to(device)  # [b, 1, h, w]
    elif mode == 'laplacian':
        # NOTE: Laplacian TODO
        pass

    return edge


def edge_extractor_torch(x: Tensor, mode="sobel", device="cuda:0"):
    conv_rgb_core_sobel_horizontal = [
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1],
         [0, 0, 0], [0, 0, 0], [0, 0, 0],
         [0, 0, 0], [0, 0, 0], [0, 0, 0]
         ],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0],
         [1, 2, 1], [0, 0, 0], [-1, -2, -1],
         [0, 0, 0], [0, 0, 0], [0, 0, 0]
         ],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0],
         [0, 0, 0], [0, 0, 0], [0, 0, 0],
         [1, 2, 1], [0, 0, 0], [-1, -2, -1],
         ]]
    conv_rgb_core_sobel_vertical = [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
         [0, 0, 0], [0, 0, 0], [0, 0, 0],
         [0, 0, 0], [0, 0, 0], [0, 0, 0]
         ],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0],
         [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
         [0, 0, 0], [0, 0, 0], [0, 0, 0]
         ],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0],
         [0, 0, 0], [0, 0, 0], [0, 0, 0],
         [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
         ]]

    conv_op_horizontal = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
    conv_op_vertical = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)

    sobel_kernel_horizontal = np.array(conv_rgb_core_sobel_horizontal, dtype='float32')
    sobel_kernel_horizontal = sobel_kernel_horizontal.reshape((3, 3, 3, 3))
    conv_op_horizontal.weight.data = torch.from_numpy(sobel_kernel_horizontal).to(device)

    sobel_kernel_vertical = np.array(conv_rgb_core_sobel_vertical, dtype='float32')
    sobel_kernel_vertical = sobel_kernel_vertical.reshape((3, 3, 3, 3))
    conv_op_vertical.weight.data = torch.from_numpy(sobel_kernel_vertical).to(device)

    sobel_x = conv_op_horizontal(x)
    sobel_y = conv_op_vertical(x)
    sobel_xy = torch.sqrt(torch.square(sobel_x) + torch.square(sobel_y))

    sobel_x = normalize2img_tensor(sobel_x)
    sobel_y = normalize2img_tensor(sobel_y)
    sobel_xy = normalize2img_tensor(sobel_xy)

    return sobel_x, sobel_y, sobel_xy


class GESA(nn.Module):
    """
    Guided edge spatial attention module from "Discrete Cosine Transform Network for Guided Depth Map Super-Resolution"
    """
    def __init__(self, n_feats=64):
        super(GESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)  # [b, c, (h-1)/2, (w-1)/2]
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv_dilation = nn.Conv2d(f, f, kernel_size=3, padding=1, stride=3, dilation=2)

    def forward(self, x):  # x is the input feature
        x = self.conv1(x)
        shortCut = x
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=7, stride=3)  # [b, c, (h-4)/3, (w-4)/3]
        x = self.relu(self.conv_max(x))
        x = self.relu(self.conv3(x))
        x = self.conv3_(x)
        x = F.interpolate(x, (shortCut.size(2), shortCut.size(3)), mode='bilinear', align_corners=False)
        shortCut = self.conv_f(shortCut)
        x = self.conv4(x + shortCut)
        x = self.sigmoid(x)
        return x


# class DDRNetEdge(nn.Module):
#     def __init__(self,
#                  opts,
#                  pretrained=True,
#                  up_features=[64, 32, 16],
#                  inner_features=[64, 32, 16]):
#         super(DDRNetEdge, self).__init__()
#
#         self.bs = getattr(opts, "common.bs", 8)
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         # self.edge_batch_tensor = torch.randn(size=(self.bs, 1, 480, 640))
#
#         self.feature_extractor = DualResNet_Backbone(
#             pretrained=pretrained,
#             features=up_features[0])
#
#         # self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
#         #                                     expand_features=inner_features[0],
#         #                                     out_features=up_features[1],
#         #                                     kernel_size=3,
#         #                                     channel_attention=True,
#         #                                     guide_features=3,
#         #                                     guidance_type="full")
#         # self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
#         #                                     expand_features=inner_features[1],
#         #                                     out_features=up_features[2],
#         #                                     kernel_size=3,
#         #                                     channel_attention=True,
#         #                                     guide_features=3,
#         #                                     guidance_type="full")
#         # self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
#         #                                     expand_features=inner_features[2],
#         #                                     out_features=1,
#         #                                     kernel_size=3,
#         #                                     channel_attention=True,
#         #                                     guide_features=3,
#         #                                     guidance_type="full")
#
#         self.edge_encoder_1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )  # [b, 64, h/2, w/2]
#
#         self.edge_encoder_2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )  # [b, 128, h/4, w/4]
#
#         self.edge_encoder_3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )  # [b, 256, h/8, w/8]
#
#         self.btlnck_conv = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
#
#         self.upsample_1 = BaseUpsamplingBlock(in_features=64, out_features=128)
#         self.upsample_2 = BaseUpsamplingBlock(128, 64)
#         self.upsample_3 = BaseUpsamplingBlock(64, 32)
#
#         self.depth_head = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x, curr_epoch, iter_step):
#         # NOTE: RGB的context信息提取
#         y = self.feature_extractor(x)  # [b, 64, 60, 80]
#         # print("y.shape:", y.shape)
#
#         # NOTE: edge信息提取
#         b, c, h, w = x.shape
#         x_ = x
#         x_ = x_ * 255
#         x_ = x_.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
#         # x_ = x.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)  # [b, h, w, c]
#
#         # print("x_.shape:", x_.shape)
#         # print("x_[0].shape:", x_[0].shape)
#
#         # temp = []
#         # for i in range(b):
#         #     canny_edge = cv.Canny(x_[i, :, :, :], 100, 200)
#         #     canny_edge = np.expand_dims(canny_edge, axis=0)  # [1, h, w]
#         #     canny_edge = np.expand_dims(canny_edge, axis=0)  # [1, 1, h, w]
#         #     temp.append(torch.from_numpy(canny_edge).type(torch.float32))
#         # edge = torch.cat(temp, dim=0).to(self.device)  # [b, 1, h, w]
#         # edge = torch.clamp(edge, min=0., max=1.)  # 不是重新映射，是在给定区间处截断
#
#         # temp = []
#         edge_batch_tensor = torch.randn(size=(b, 1, h, w))
#         for i in range(b):
#             canny_edge = cv.Canny(x_[i, :, :, :], 100, 200)
#             canny_edge = np.expand_dims(canny_edge, axis=0)  # [1, h, w]
#             # temp.append(canny_edge.transpose(1, 2, 0))
#             canny_edge = np.expand_dims(canny_edge, axis=0)  # [1, 1, h, w]
#             canny_edge = normalize(canny_edge)  # 将数据缩放到[0, 1]区间
#             # self.edge_batch_tensor[i, :, :, :] = torch.from_numpy(canny_edge).type(torch.float32)
#             edge_batch_tensor[i, :, :, :] = torch.from_numpy(canny_edge).type(torch.float32)
#         # edge = self.edge_batch_tensor.to(self.device)  # [b, 1, h, w]
#         edge = edge_batch_tensor.to(self.device)  # [b, 1, h, w]
#
#         if curr_epoch == '0':
#             if iter_step >= (50688 // self.bs) - 2:
#                 print("epoch:{}, iter_step:{}, y:{}, x:{}, edge:{}"
#                       .format(curr_epoch, iter_step, y.shape, x.shape, edge.shape))
#         if curr_epoch == '1':
#             print("epoch:{}, iter_step:{}, y:{}, x:{}, edge:{}"
#                   .format(curr_epoch, iter_step, y.shape, x.shape, edge.shape))
#
#         # cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
#         # cmap_type = cmap_type_list[1]
#         # for i in range(b):
#         #     plt.subplot(2, 2, i + 1)
#         #     plt.imshow(x_[i])  # , cmap=cmap_type
#         #     plt.subplot(2, 2, i + 3)
#         #     plt.imshow(temp[i])
#         # plt.show()
#
#         #     cv.imshow("img_{}".format(i), x_[i])
#         # cv.waitKey(0)
#         # cv.destroyAllWindows()
#
#
#         edge1 = self.edge_encoder_1(edge)  # [b, 64, h/2, w/2]
#         edge2 = self.edge_encoder_2(edge1)  # [b, 128, h/4, w/4]
#         edge3 = self.edge_encoder_3(edge2)  # [b, 256, h/8, w/8]
#         # print("edge1.shape:", edge1.shape)
#         # print("edge2.shape:", edge2.shape)
#         # print("edge3.shape:", edge3.shape)
#         # print("min:{}, max:{}".format(torch.min(edge), torch.max(edge)))
#
#         edge3 = self.btlnck_conv(edge3)  # [b, 64, h/8, w/8]
#         # print("edge3.shape:", edge3.shape)
#         y = self.upsample_1(y, edge3)  # [b, 128, h/4, w/4]
#         # print("y.shape:", y.shape)
#         y = self.upsample_2(y, edge2)  # [b, 64, h/2, w/2]
#         # print("y.shape:", y.shape)
#         y = self.upsample_3(y, edge1)  # [b, 32, h, w]
#         # print("y.shape:", y.shape)
#
#         y = self.depth_head(y)  # [b, 1, h, w]
#         # print("y.shape:", y.shape)
#
#         return y
#
#
#
#         # x_half = F.interpolate(x, scale_factor=.5)
#         # x_quarter = F.interpolate(x, scale_factor=.25)
#         #
#         # y = F.interpolate(y, scale_factor=2.0, mode='bilinear')  # [b, 64, 120, 160]
#         # y = self.up_1(x_quarter, y)
#         #
#         # y = F.interpolate(y, scale_factor=2.0, mode='bilinear')  # [b, 64, 240, 320]
#         # y = self.up_2(x_half, y)
#         #
#         # y = F.interpolate(y, scale_factor=2.0, mode='bilinear')  # [b, 64, 480, 640]
#         # y = self.up_3(x, y)
#         # return y


class SELayer(nn.Module):
    """
    Taken from:
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=[2, 3])  # Replacement of avgPool for large kernels for trt
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand(x.shape)


class EdgeNetV1(nn.Module):
    def __init__(self,
                 shared_kernel_1=None,
                 shared_kernel_2=None,
                 shared_kernel_3=None
                 ):
        super(EdgeNetV1, self).__init__()

        # Params: 1.27M
        # self.edge_encoder_1 = nn.Sequential(
        #     nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        # )  # [b, 32, h/2, w/2]
        #
        # self.edge_encoder_2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        #     nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )  # [b, 128, h/4, w/4]
        #
        # self.edge_encoder_3 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
        #     nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )  # [b, 256, h/8, w/8]

        # Params: 0.25M
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # kernel: [8, 3, 3, 3]
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # kernel: [16, 8, 3, 3]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/4, w/4]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # [b, 256, h/8, w/8]

        self.squeeze_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.squeeze_conv(self.edge_encoder_3(x))  # [B, 64, 60, 80]
        features.append(x)
        return features


class EdgeNetV1FM(nn.Module):
    """
    shallow:few layers
    narrow: few channels (feature maps)
    """
    def __init__(self):
        super(EdgeNetV1FM, self).__init__()
        # Params: 0.25M
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/4, w/4]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 32, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.edge_encoder_3(x)  # [B, 64, 60, 80]
        features.append(x)
        return features


class EdgeNetKSV1(nn.Module):
    """
    KS: partial Kernel Shared
    """
    def __init__(self,
                 shared_kernel_1=None,
                 shared_kernel_2=None,
                 shared_kernel_3=None
                 ):
        super(EdgeNetKSV1, self).__init__()

        # Params: 0.25M
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # kernel: [8, 3, 3, 3]
            nn.BatchNorm2d(8),
            # nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # kernel: [16, 8, 3, 3]
            # nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]
        self.edge_encoder_1_res = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(32),
            # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
        )  # [b, 128, h/4, w/4]
        self.edge_encoder_2_res = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/4, w/4]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(128),
            # nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )  # [b, 256, h/8, w/8]
        self.edge_encoder_3_res = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # [b, 256, h/8, w/8]

        self.squeeze_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

        # # NOTE: 共享kernel的输入通道数需要保持一致，在输出通道上联结
        # shared_kernel_1[:8, :8, :, :]  # [32, 32, 3, 3]->[8, 8, 3, 3]
        # shared_kernel_2[:32, :32, :, :]  # [64, 64, 3, 3]->[32, 32, 3, 3]
        # shared_kernel_3  # [64, 128, 3, 3]

        self.kernel_1 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[8, 8, 3, 3])))
        self.kernel_2 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[32, 32, 3, 3])))
        self.kernel_3 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[64, 128, 3, 3])))

    def forward(self, x, shared_kernel_1=None, shared_kernel_2=None, shared_kernel_3=None):
        # NOTE: 共享kernel的输入通道数需要保持一致，在输出通道上联结
        # shared_kernel_1[:8, :8, :, :]  # [32, 32, 3, 3]->[8, 8, 3, 3]
        # shared_kernel_2[:32, :32, :, :]  # [64, 64, 3, 3]->[32, 32, 3, 3]
        # shared_kernel_3  # [64, 128, 3, 3]

        features = []
        x = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        x = F.conv2d(x, torch.cat((shared_kernel_1[:8, :8, :, :], self.kernel_1), dim=0), stride=2, padding=1)
        x = self.edge_encoder_1_res(x)
        features.append(x)

        x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
        x = F.conv2d(x, torch.cat((shared_kernel_2[:32, :32, :, :], self.kernel_2), dim=0), stride=2, padding=1)
        x = self.edge_encoder_2_res(x)
        features.append(x)

        x = self.edge_encoder_3(x)
        x = F.conv2d(x, torch.cat((shared_kernel_3, self.kernel_3), dim=0), stride=2, padding=1)
        x = self.edge_encoder_3_res(x)
        x = self.squeeze_conv(x)  # [B, 64, 60, 80]
        features.append(x)

        return features


class EdgeNetV2(nn.Module):
    """
    将普通的卷积替换成了深度可分离卷积
    """
    def __init__(self):
        super(EdgeNetV2, self).__init__()

        # Params: 0.25M
        self.edge_encoder_1 = nn.Sequential(
            # nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # kernel: [8, 3, 3, 3]
            nn.Conv2d(3, (3 * 3), kernel_size=3, stride=1, padding=1, groups=3),  # kernel: [8, 3, 3, 3]
            nn.BatchNorm2d(9),
            # nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # kernel: [16, 8, 3, 3]
            nn.Conv2d(9, (9 * 2), kernel_size=3, stride=2, padding=1, groups=9),  # kernel: [16, 8, 3, 3]
            nn.BatchNorm2d(18),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(18, (18 * 2), kernel_size=3, stride=1, padding=1, groups=18),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(36),
            nn.Conv2d(36, (36 * 2), kernel_size=3, stride=2, padding=1, groups=36),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/4, w/4]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(72, (72 * 2), kernel_size=3, stride=1, padding=1, groups=72),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(144),
            nn.Conv2d(144, (144 * 1), kernel_size=3, stride=2, padding=1, groups=144),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True),
        )  # [b, 256, h/8, w/8]

        self.squeeze_conv = nn.Conv2d(144, 72, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 18, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 72, 120, 160]
        features.append(x)
        x = self.squeeze_conv(self.edge_encoder_3(x))  # [B, 144, 60, 80]
        features.append(x)
        return features


class EdgeNetV3(nn.Module):
    """
    使用膨胀卷积替换常规卷积
    """
    def __init__(self):
        super(EdgeNetV3, self).__init__()
        # Params: 0.25M
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, dilation=1),  # kernel: [8, 3, 3, 3]
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, dilation=1),  # kernel: [16, 8, 3, 3]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2, dilation=2),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/4, w/4]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3, dilation=3),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=3, dilation=3),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # [b, 256, h/8, w/8]

        self.squeeze_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.squeeze_conv(self.edge_encoder_3(x))  # [B, 64, 60, 80]
        features.append(x)
        return features


# class Decoder(nn.Module):
#     def __init__(self, bs):
#         super(Decoder, self).__init__()
#
#         # NOTE: 上采样可以用transpose_conv/unpooling+conv来完成
#         self.conv_1_1 = nn.Sequential(
#             nn.Conv2d(64+64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.SE_block_1 = SELayer(64, reduction=1)
#         self.conv_1_2 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv_2_1 = nn.Sequential(
#             nn.Conv2d(64+64+32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.SE_block_2 = SELayer(64, reduction=1)
#         self.conv_2_2 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )
#
#         self.conv_3_1 = nn.Sequential(
#             nn.Conv2d(32+16+32, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )
#         self.SE_block_3 = SELayer(32, reduction=1)
#         self.conv_3_2 = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, features, edges, x=None):
#         x1 = torch.cat((features[-1], edges[-1]), dim=1)
#         x1 = self.conv_1_1(x1)
#         x1 = self.SE_block_1(x1)
#         x1 = x1 + features[-1]
#         x1 = self.conv_1_2(x1)
#         x1 = F.interpolate(x1, scale_factor=2.0, mode='bilinear')  # [B, 32, 120, 160]
#
#         features[1] = F.interpolate(features[1], scale_factor=2.0, mode='bilinear')  # [B, 64, 120, 160]
#         x2 = torch.cat((features[1], edges[-2], x1), dim=1)  # [B, 64+64+32, 120, 160]
#         x2 = self.conv_2_1(x2)
#         x2 = self.SE_block_2(x2)
#         x2 = x2 + features[1]
#         x2 = self.conv_2_2(x2)
#         x2 = F.interpolate(x2, scale_factor=2.0, mode='bilinear')  # [B, 32, 240, 320]
#
#         features[0] = F.interpolate(features[0], scale_factor=2.0, mode='bilinear')  # [B, 32, 240, 320]
#         x3 = torch.cat((features[0], edges[-3], x2), dim=1)  # [B, 32+16+32, 240, 320]
#         x3 = self.conv_3_1(x3)
#         x3 = self.SE_block_3(x3)
#         x3 = x3 + features[0]
#         x3 = F.interpolate(x3, scale_factor=2.0, mode='bilinear')  # [B, 32, 480, 640]
#         x3 = self.conv_3_2(x3)
#
#         return x3


class DecoderBlockV1(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_cbam=True,
                 is_lastblock=False):
        super(DecoderBlockV1, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam
        self.is_lastblock = is_lastblock

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        if use_cbam is True:
            self.cbam = CBAM(n_feats)

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, rgb_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(rgb_feats),
            nn.ReLU(inplace=True),
        )
        self.SE_block = SELayer(rgb_feats, reduction=1)

        if is_lastblock is False:
            self.conv2 = nn.Sequential(
                nn.Conv2d(rgb_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(rgb_feats, rgb_feats//2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(rgb_feats//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(rgb_feats//2, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, rgb_feature, edge, target=None):
        if self.block_sn >= 2:
            rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        if self.use_cbam is True:
            x = self.cbam(x)

        x = self.SE_block(self.conv1(x))
        x = x + rgb_feature

        if self.is_lastblock is False:
            x = self.conv2(x)
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            x = self.conv2(x)

        return x


class DecoderV1_(nn.Module):
    def __init__(self):
        super(DecoderV1_, self).__init__()

        self.decoder_block_1 = DecoderBlockV1(1, 64, 64, output_feats=32)
        self.decoder_block_2 = DecoderBlockV1(2, 64, 64, output_feats=32, target_feats=32)
        self.decoder_block_3 = DecoderBlockV1(3, 32, 16, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[1], edges[-2], target=y)
        y = self.decoder_block_3(features[0], edges[-3], target=y)

        return y


class DecoderV1(nn.Module):
    def __init__(self, use_cbam=True):
        super(DecoderV1, self).__init__()

        self.use_cbam = use_cbam

        # NOTE: 上采样可以用transpose_conv/unpooling+conv来完成
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.SE_block_1 = SELayer(64, reduction=1)
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(64+64+32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.SE_block_2 = SELayer(64, reduction=1)
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv_3_1 = nn.Sequential(
            # nn.Conv2d(32+16+32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32 + 16 + 32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.SE_block_3 = SELayer(32, reduction=1)
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
        )

        # self.cbam1 = CBAM(bs * 128)  # bs*c=8*128 Params:0.14M
        # self.cbam2 = CBAM(bs * 160)  # 8*128 Params:0.175M
        # self.cbam3 = CBAM(bs * 80)  # 8*128 Params:0.14M

        if use_cbam:
            self.cbam1 = CBAM(128)  # Params: M
            self.cbam2 = CBAM(160)  # Params: M
            self.cbam3 = CBAM(80)  # Params: M
            # self.cbam3 = CBAM(96)  # Params: M

    def forward(self, features, edges, x=None):
        before_add = []
        after_add = []
        x1 = torch.cat((features[-1], edges[-1]), dim=1)
        if self.use_cbam:
            x1 = self.cbam1(x1)
        x1 = self.conv_1_1(x1)
        x1 = self.SE_block_1(x1)
        before_add.append(x1)
        x1 = x1 + features[-1]
        after_add.append(x1)
        x1 = self.conv_1_2(x1)
        x1 = F.interpolate(x1, scale_factor=2.0, mode='bilinear')  # [B, 32, 120, 160]

        features[1] = F.interpolate(features[1], scale_factor=2.0, mode='bilinear')  # [B, 64, 120, 160]
        x2 = torch.cat((features[1], edges[-2], x1), dim=1)  # [B, 64+64+32, 120, 160]
        if self.use_cbam:
            x2 = self.cbam2(x2)
        x2 = self.conv_2_1(x2)
        x2 = self.SE_block_2(x2)
        before_add.append(x2)
        x2 = x2 + features[1]
        after_add.append(x2)
        x2 = self.conv_2_2(x2)
        x2 = F.interpolate(x2, scale_factor=2.0, mode='bilinear')  # [B, 32, 240, 320]

        features[0] = F.interpolate(features[0], scale_factor=2.0, mode='bilinear')  # [B, 32, 240, 320]
        x3 = torch.cat((features[0], edges[-3], x2), dim=1)  # [B, 32+16+32, 240, 320]
        if self.use_cbam:
            x3 = self.cbam3(x3)
        x3 = self.conv_3_1(x3)
        x3 = self.SE_block_3(x3)
        before_add.append(x3)
        x3 = x3 + features[0]
        after_add.append(x3)
        x3 = F.interpolate(x3, scale_factor=2.0, mode='bilinear')  # [B, 32, 480, 640]
        x3 = self.conv_3_2(x3)

        return x3  # x3, before_add, after_add


class DecoderV1PConv(nn.Module):
    def __init__(self, use_cbam=True):
        super(DecoderV1PConv, self).__init__()

        drop_path_rate = 0.1
        # 用到了6个FasterMutableBlock drop概率逐步提升
        dpr = [x.item() for x in torch.linspace(0, end=drop_path_rate, steps=6)]

        self.use_cbam = use_cbam

        # NOTE: 上采样可以用transpose_conv/unpooling+conv来完成
        self.conv_1_1 = nn.Sequential(
            # nn.Conv2d(64+64, 64, kernel_size=3, stride=1, padding=1),
            FasterMutableBlock(dim=64+64, out_dim=64, drop_path=dpr[0]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.SE_block_1 = SELayer(64, reduction=1)
        self.conv_1_2 = nn.Sequential(
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            FasterMutableBlock(dim=64, out_dim=32, drop_path=dpr[1]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv_2_1 = nn.Sequential(
            # nn.Conv2d(64+64+32, 64, kernel_size=3, stride=1, padding=1),
            FasterMutableBlock(dim=64+64+32, out_dim=64, drop_path=dpr[2]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.SE_block_2 = SELayer(64, reduction=1)
        self.conv_2_2 = nn.Sequential(
            # nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            FasterMutableBlock(dim=64, out_dim=32, drop_path=dpr[3]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv_3_1 = nn.Sequential(
            # nn.Conv2d(32+16+32, 32, kernel_size=3, stride=1, padding=1),
            FasterMutableBlock(dim=32+16+32, out_dim=32, drop_path=dpr[4]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.SE_block_3 = SELayer(32, reduction=1)
        self.conv_3_2 = nn.Sequential(
            # nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            FasterMutableBlock(dim=32, out_dim=16, drop_path=dpr[5]),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
        )

        # self.cbam1 = CBAM(bs * 128)  # bs*c=8*128 Params:0.14M
        # self.cbam2 = CBAM(bs * 160)  # 8*128 Params:0.175M
        # self.cbam3 = CBAM(bs * 80)  # 8*128 Params:0.14M

        if use_cbam:
            self.cbam1 = CBAM(128)  # Params: M
            self.cbam2 = CBAM(160)  # Params: M
            self.cbam3 = CBAM(80)  # Params: M

    def forward(self, features, edges, x=None):
        x1 = torch.cat((features[-1], edges[-1]), dim=1)
        if self.use_cbam:
            x1 = self.cbam1(x1)
        x1 = self.conv_1_1(x1)
        x1 = self.SE_block_1(x1)
        x1 = x1 + features[-1]
        x1 = self.conv_1_2(x1)
        x1 = F.interpolate(x1, scale_factor=2.0, mode='bilinear')  # [B, 32, 120, 160]

        features[1] = F.interpolate(features[1], scale_factor=2.0, mode='bilinear')  # [B, 64, 120, 160]
        x2 = torch.cat((features[1], edges[-2], x1), dim=1)  # [B, 64+64+32, 120, 160]
        if self.use_cbam:
            x2 = self.cbam2(x2)
        x2 = self.conv_2_1(x2)
        x2 = self.SE_block_2(x2)
        x2 = x2 + features[1]
        x2 = self.conv_2_2(x2)
        x2 = F.interpolate(x2, scale_factor=2.0, mode='bilinear')  # [B, 32, 240, 320]

        features[0] = F.interpolate(features[0], scale_factor=2.0, mode='bilinear')  # [B, 32, 240, 320]
        x3 = torch.cat((features[0], edges[-3], x2), dim=1)  # [B, 32+16+32, 240, 320]
        if self.use_cbam:
            x3 = self.cbam3(x3)
        x3 = self.conv_3_1(x3)
        x3 = self.SE_block_3(x3)
        x3 = x3 + features[0]
        x3 = F.interpolate(x3, scale_factor=2.0, mode='bilinear')  # [B, 32, 480, 640]
        x3 = self.conv_3_2(x3)

        return x3


class DecoderBlockKSV1(nn.Module):
    def __init__(self,
                 target_feats,
                 edge_feats,
                 output_feats):
        super(DecoderBlockKSV1, self).__init__()

        n_feats = target_feats + edge_feats
        shared_feats = n_feats // 2

        self.rgb_init_conv = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )

        self.cbam = CBAM(n_feats)

        self.kernel_rgb_1 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[n_feats-shared_feats, n_feats, 3, 3])))
        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[shared_feats, n_feats, 3, 3])))
        self.kernel_fuse_1 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[n_feats-shared_feats, n_feats, 3, 3])))
        self.bias_rgb_1 = nn.Parameter((torch.zeros(size=[n_feats-shared_feats])))
        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[shared_feats])))
        self.bias_fuse_1 = nn.Parameter((torch.zeros(size=[n_feats-shared_feats])))
        self.rgb_bn_acti_1 = nn.Sequential(
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )
        self.fuse_bn_acti_1 = nn.Sequential(
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )

        self.kernel_rgb_2 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[n_feats-shared_feats, n_feats, 3, 3])))
        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[shared_feats, n_feats, 3, 3])))
        self.kernel_fuse_2 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[n_feats-shared_feats, n_feats, 3, 3])))
        self.bias_rgb_2 = nn.Parameter((torch.zeros(size=[n_feats-shared_feats])))
        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[shared_feats])))
        self.bias_fuse_2 = nn.Parameter((torch.zeros(size=[n_feats-shared_feats])))
        self.rgb_bn_acti_2 = nn.Sequential(
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )
        self.fuse_bn_acti_2 = nn.Sequential(
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )

        self.se_block = SELayer(n_feats * 2, reduction=1)

        self.output_conv = nn.Sequential(
            nn.Conv2d(n_feats * 2, output_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )

    def forward(self, target, edge, raw_rgb):
        x = self.rgb_init_conv(raw_rgb)
        shortcut_x = x

        x = F.conv2d(
            x,
            torch.cat([self.kernel_rgb_1, self.kernel_shared_1], dim=0),
            torch.cat([self.bias_rgb_1, self.bias_shared_1], dim=0),
            stride=1,
            padding=1
        )
        x = self.rgb_bn_acti_1(x)
        x = F.conv2d(
            x,
            torch.cat([self.kernel_rgb_2, self.kernel_shared_2], dim=0),
            torch.cat([self.bias_rgb_2, self.bias_shared_2], dim=0),
            stride=1,
            padding=1
        )
        x = self.rgb_bn_acti_2(x)
        x = x + shortcut_x

        y = torch.cat((target, edge), dim=1)
        y = self.cbam(y)
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')
        shortcut_y = y

        y = F.conv2d(
            y,
            torch.cat([self.kernel_fuse_1, self.kernel_shared_1], dim=0),
            torch.cat([self.bias_fuse_1, self.bias_shared_1], dim=0),
            stride=1,
            padding=1
        )
        y = self.fuse_bn_acti_1(y)
        y = F.conv2d(
            y,
            torch.cat([self.kernel_fuse_2, self.kernel_shared_2], dim=0),
            torch.cat([self.bias_fuse_2, self.bias_shared_2], dim=0),
            stride=1,
            padding=1
        )
        y = self.fuse_bn_acti_2(y)
        y = y + shortcut_y

        xy = torch.cat((x, y), dim=1)
        xy = self.se_block(xy)
        xy = self.output_conv(xy)
        return xy


class DecoderKSV1(nn.Module):
    """
    KS: partial Kernel Shared
    """
    def __init__(self):
        super(DecoderKSV1, self).__init__()

        self.block1 = DecoderBlockKSV1(target_feats=64, edge_feats=64, output_feats=64)
        self.block2 = DecoderBlockKSV1(target_feats=64, edge_feats=64, output_feats=16)
        self.block3 = DecoderBlockKSV1(target_feats=16, edge_feats=16, output_feats=1)

    def forward(self, features, edges, x):
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        target2 = self.block1(features[-1], edges[-1], x_4)  # [B, 64, 120, 160]
        target3 = self.block2(target2, edges[-2], x_2)  # [B, 16, 240, 320]
        target_final = self.block3(target3, edges[-3], x)  # [B, 1, 480, 640]

        return target_final


class DecoderBlockKSV2(nn.Module):
    def __init__(self,
                 target_feats,
                 edge_feats,  # 默认edge_feats=target_feats
                 output_feats,
                 use_conv1x1=True
                 ):
        super(DecoderBlockKSV2, self).__init__()

        n_feats = target_feats
        shared_feats = n_feats // 2
        self.use_conv1x1 = use_conv1x1

        self.kernel_rgb_1 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[n_feats-shared_feats, n_feats, 3, 3])))
        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[shared_feats, n_feats, 3, 3])))
        self.kernel_edge_1 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[n_feats-shared_feats, n_feats, 3, 3])))
        self.bias_rgb_1 = nn.Parameter((torch.zeros(size=[n_feats-shared_feats])))
        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[shared_feats])))
        self.bias_edge_1 = nn.Parameter((torch.zeros(size=[n_feats-shared_feats])))
        self.rgb_bn_acti_1 = nn.Sequential(
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )
        self.edge_bn_acti_1 = nn.Sequential(
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )

        self.kernel_rgb_2 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[n_feats-shared_feats, n_feats, 3, 3])))
        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[shared_feats, n_feats, 3, 3])))
        self.kernel_edge_2 = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(size=[n_feats-shared_feats, n_feats, 3, 3])))
        self.bias_rgb_2 = nn.Parameter((torch.zeros(size=[n_feats-shared_feats])))
        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[shared_feats])))
        self.bias_edge_2 = nn.Parameter((torch.zeros(size=[n_feats-shared_feats])))
        self.rgb_bn_acti_2 = nn.Sequential(
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )
        self.edge_bn_acti_2 = nn.Sequential(
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )

        # self.se_block = SELayer(n_feats * 2, reduction=1)

        if use_conv1x1:
            self.conv_1x1 = nn.Conv2d(target_feats+edge_feats, n_feats, kernel_size=1, stride=1, padding=0)
            self.cbam = CBAM(n_feats)
            self.output_conv = nn.Sequential(
                nn.Conv2d(n_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.cbam = CBAM(target_feats+edge_feats)
            self.output_conv = nn.Sequential(
                nn.Conv2d(target_feats+edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )

    def forward(self, target, edge, rgb_features):
        target = F.interpolate(target, scale_factor=2.0, mode='bilinear')
        edge = F.interpolate(edge, scale_factor=2.0, mode='bilinear')

        shortcut_x = target
        x = F.conv2d(
            target,
            torch.cat([self.kernel_rgb_1, self.kernel_shared_1], dim=0),
            torch.cat([self.bias_rgb_1, self.bias_shared_1], dim=0),
            stride=1,
            padding=1
        )
        x = self.rgb_bn_acti_1(x)
        x = F.conv2d(
            x,
            torch.cat([self.kernel_rgb_2, self.kernel_shared_2], dim=0),
            torch.cat([self.bias_rgb_2, self.bias_shared_2], dim=0),
            stride=1,
            padding=1
        )
        x = self.rgb_bn_acti_2(x)
        x = x + shortcut_x

        shortcut_y = edge
        y = F.conv2d(
            edge,
            torch.cat([self.kernel_edge_1, self.kernel_shared_1], dim=0),
            torch.cat([self.bias_edge_1, self.bias_shared_1], dim=0),
            stride=1,
            padding=1
        )
        y = self.edge_bn_acti_1(y)
        y = F.conv2d(
            y,
            torch.cat([self.kernel_edge_2, self.kernel_shared_2], dim=0),
            torch.cat([self.bias_edge_2, self.bias_shared_2], dim=0),
            stride=1,
            padding=1
        )
        y = self.edge_bn_acti_2(y)
        y = y + shortcut_y

        xy = torch.cat((x, y), dim=1)
        if self.use_conv1x1:
            xy = self.conv_1x1(xy)
        xy = self.cbam(xy)
        xy = xy + rgb_features
        xy = self.output_conv(xy)

        return xy


class DecoderKSV2(nn.Module):
    def __init__(self):
        super(DecoderKSV2, self).__init__()

        self.block1 = DecoderBlockKSV2(target_feats=64, edge_feats=64, output_feats=64)
        self.block2 = DecoderBlockKSV2(target_feats=64, edge_feats=64, output_feats=16)
        self.block3 = DecoderBlockKSV2(target_feats=16, edge_feats=16, output_feats=1, use_conv1x1=False)

    def forward(self, features, edges, x=None):
        rgb_features1 = F.interpolate(features[-1], scale_factor=2.0, mode='bilinear')
        rgb_features2 = F.interpolate(features[1], scale_factor=4.0, mode='bilinear')
        rgb_features3 = F.interpolate(features[0], scale_factor=4.0, mode='bilinear')

        # target, edge, rgb_features
        target2 = self.block1(features[-1], edges[-1], rgb_features1)
        target3 = self.block2(target2, edges[-2], rgb_features2)
        target_final = self.block3(target3, edges[-3], rgb_features3)

        return target_final


class DecoderBlockV3(nn.Module):
    def __init__(self,
                 target_feats,
                 edge_feats,  # 默认edge_feats=target_feats
                 output_feats,
                 use_conv1x1=True
                 ):
        super(DecoderBlockV3, self).__init__()

        self.use_conv1x1 = use_conv1x1
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(target_feats, target_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(target_feats*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_feats * 2, target_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(target_feats),
            nn.ReLU(inplace=True),
        )

        self.edge_conv = nn.Sequential(
            nn.Conv2d(edge_feats, edge_feats * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_feats * 2, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        if use_conv1x1:
            self.conv_1x1 = nn.Conv2d(target_feats+edge_feats, target_feats, kernel_size=1, stride=1, padding=0)
            self.cbam = CBAM(target_feats)
            self.output_conv = nn.Sequential(
                nn.Conv2d(target_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.cbam = CBAM(target_feats+edge_feats)
            self.output_conv = nn.Sequential(
                nn.Conv2d(target_feats+edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )

    def forward(self, target, edge, rgb_features):
        target = F.interpolate(target, scale_factor=2.0, mode='bilinear')
        edge = F.interpolate(edge, scale_factor=2.0, mode='bilinear')

        shortcut_x = target
        x = self.rgb_conv(target)
        x = x + shortcut_x

        shortcut_y = edge
        y = self.edge_conv(edge)
        y = y + shortcut_y

        xy = torch.cat((x, y), dim=1)
        if self.use_conv1x1:
            xy = self.conv_1x1(xy)
        xy = self.cbam(xy)
        xy = xy + rgb_features
        xy = self.output_conv(xy)

        return xy


class DecoderV3(nn.Module):
    def __init__(self):
        super(DecoderV3, self).__init__()

        self.block1 = DecoderBlockV3(target_feats=64, edge_feats=64, output_feats=64)
        self.block2 = DecoderBlockV3(target_feats=64, edge_feats=64, output_feats=16)
        self.block3 = DecoderBlockV3(target_feats=16, edge_feats=16, output_feats=1, use_conv1x1=False)

    def forward(self, features, edges, x=None):
        rgb_features1 = F.interpolate(features[-1], scale_factor=2.0, mode='bilinear')
        rgb_features2 = F.interpolate(features[1], scale_factor=4.0, mode='bilinear')
        rgb_features3 = F.interpolate(features[0], scale_factor=4.0, mode='bilinear')

        # target, edge, rgb_features
        target2 = self.block1(features[-1], edges[-1], rgb_features1)
        target3 = self.block2(target2, edges[-2], rgb_features2)
        target_final = self.block3(target3, edges[-3], rgb_features3)

        return target_final


class DecoderV4(nn.Module):
    def __init__(self, use_cbam=False):
        super(DecoderV4, self).__init__()

        self.use_cbam = use_cbam

        # NOTE: 上采样可以用transpose_conv/unpooling+conv来完成
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.SE_block_1 = SELayer(64, reduction=1)
        self.conv_1_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid()
        )

        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(64+64+32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.SE_block_2 = SELayer(64, reduction=1)
        self.conv_2_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Sigmoid()
        )

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(32+16+32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.SE_block_3 = SELayer(32, reduction=1)
        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
        )
        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        if use_cbam:
            self.cbam1 = CBAM(128)  # Params: M
            self.cbam2 = CBAM(160)  # Params: M
            self.cbam3 = CBAM(80)  # Params: M

    def forward(self, features, edges, x=None):
        x1 = torch.cat((features[-1], edges[-1]), dim=1)
        if self.use_cbam:
            x1 = self.cbam1(x1)
        x1 = self.conv_1_1(x1)
        x1 = self.SE_block_1(x1)
        x1 = x1 + features[-1]
        x1 = self.conv_1_2(x1)
        x1 = self.deconv_1(x1)  # [B, 32, 120, 160]

        features[1] = F.interpolate(features[1], scale_factor=2.0, mode='bilinear')  # [B, 64, 120, 160]
        x2 = torch.cat((features[1], edges[-2], x1), dim=1)  # [B, 64+64+32, 120, 160]
        if self.use_cbam:
            x2 = self.cbam2(x2)
        x2 = self.conv_2_1(x2)
        x2 = self.SE_block_2(x2)
        x2 = x2 + features[1]
        x2 = self.conv_2_2(x2)
        x2 = self.deconv_2(x2)  # [B, 32, 240, 320]

        features[0] = F.interpolate(features[0], scale_factor=2.0, mode='bilinear')  # [B, 32, 240, 320]
        x3 = torch.cat((features[0], edges[-3], x2), dim=1)  # [B, 32+16+32, 240, 320]
        if self.use_cbam:
            x3 = self.cbam3(x3)
        x3 = self.conv_3_1(x3)
        x3 = self.SE_block_3(x3)
        x3 = x3 + features[0]
        x3 = self.conv_3_2(x3)
        x3 = self.deconv_3(x3)  # [B, 1, 480, 640]

        return x3


class DecoderBlockV5(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_cbam=True,
                 is_lastblock=False):
        super(DecoderBlockV5, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam
        self.is_lastblock = is_lastblock

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        if use_cbam is True:
            self.cbam = CBAM(edge_feats)

        if is_lastblock is False:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, edge_feats//2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(edge_feats//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_feats//2, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, rgb_feature, edge, target=None):
        if self.block_sn >= 2:
            rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)
        if self.use_cbam is True:
            x = self.cbam(x)

        # if self.block_sn == 3:
        #     temp1 = x

        x = x + edge

        # if self.block_sn == 3:
        #     temp2 = x

        if self.is_lastblock is False:
            x = self.conv2(x)
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            return x
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            x = self.conv2(x)
            return x  # x, temp1, temp2


class DecoderV5(nn.Module):
    def __init__(self):
        super(DecoderV5, self).__init__()

        self.decoder_block_1 = DecoderBlockV5(1, 64, 64, output_feats=32)
        self.decoder_block_2 = DecoderBlockV5(2, 64, 64, output_feats=32, target_feats=32)
        self.decoder_block_3 = DecoderBlockV5(3, 32, 16, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[1], edges[-2], target=y)
        # y, temp1, temp2 = self.decoder_block_3(features[0], edges[-3], target=y)
        y = self.decoder_block_3(features[0], edges[-3], target=y)

        return y  # y, temp1, temp2


class DecoderBlockV6(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_cbam=True,
                 is_lastblock=False):
        super(DecoderBlockV6, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam
        self.is_lastblock = is_lastblock

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        if use_cbam is True:
            self.cbam = CBAM(edge_feats)

        if is_lastblock is False:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, edge_feats//2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(edge_feats//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_feats//2, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, rgb_feature, edge, target=None):
        if self.block_sn >= 2:
            rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.use_cbam is True:
            x = self.cbam(x)

        # if self.block_sn == 3:
        #     temp1 = x

        x = x + edge

        # if self.block_sn == 3:
        #     temp2 = x

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.conv2(x)

        # if self.block_sn < 3:
        #     return x
        # else:
        #     return x  # , temp1, temp2
        return x


class DecoderV6(nn.Module):
    def __init__(self):
        super(DecoderV6, self).__init__()

        # NOTE: for EdgeNetV1, EdgeNetV3
        self.decoder_block_1 = DecoderBlockV6(1, 64, 64, output_feats=32)
        self.decoder_block_2 = DecoderBlockV6(2, 64, 64, output_feats=32, target_feats=32)
        self.decoder_block_3 = DecoderBlockV6(3, 32, 16, output_feats=1, target_feats=32, is_lastblock=True)

        # NOTE: for EdgeNetV2
        # self.decoder_block_1 = DecoderBlockV6(1, 64, 72, output_feats=32)
        # self.decoder_block_2 = DecoderBlockV6(2, 64, 72, output_feats=32, target_feats=32)
        # self.decoder_block_3 = DecoderBlockV6(3, 32, 18, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[1], edges[-2], target=y)
        y = self.decoder_block_3(features[0], edges[-3], target=y)
        # y, temp1, temp2 = self.decoder_block_3(features[0], edges[-3], target=y)

        return y  # , temp1, temp2


class DecoderBlockV7(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_cbam=True,
                 is_lastblock=False):
        super(DecoderBlockV7, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam
        self.is_lastblock = is_lastblock

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        if use_cbam is True:
            self.cbam = CBAM(edge_feats)

        if is_lastblock is False:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, edge_feats//2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(edge_feats//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_feats//2, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, rgb_feature, edge, target=None):
        if self.block_sn >= 2:
            rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.use_cbam is True:
            x = self.cbam(x)

        if self.is_lastblock is True:
            x = x + edge

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.conv2(x)

        return x


class DecoderV7(nn.Module):
    def __init__(self):
        super(DecoderV7, self).__init__()

        self.decoder_block_1 = DecoderBlockV7(1, 64, 64, output_feats=32)
        self.decoder_block_2 = DecoderBlockV7(2, 64, 64, output_feats=32, target_feats=32)
        self.decoder_block_3 = DecoderBlockV7(3, 32, 16, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[1], edges[-2], target=y)
        y = self.decoder_block_3(features[0], edges[-3], target=y)

        return y


class DecoderBlockV8(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_cbam=True,
                 is_lastblock=False):
        super(DecoderBlockV8, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam
        self.is_lastblock = is_lastblock

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        if use_cbam is True:
            self.cbam = CBAM(2 * edge_feats)

        if is_lastblock is False:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, edge_feats//2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(edge_feats//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_feats//2, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, rgb_feature, edge, target=None):
        if self.block_sn >= 2:
            rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)
        x = x + edge

        if self.use_cbam is True:
            x = self.cbam(x)

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.conv2(x)

        return x


class DecoderV8(nn.Module):
    def __init__(self):
        super(DecoderV8, self).__init__()

        self.decoder_block_1 = DecoderBlockV7(1, 64, 64, output_feats=32)
        self.decoder_block_2 = DecoderBlockV7(2, 64, 64, output_feats=32, target_feats=32)
        self.decoder_block_3 = DecoderBlockV7(3, 32, 16, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[1], edges[-2], target=y)
        y = self.decoder_block_3(features[0], edges[-3], target=y)

        return y


class DecoderBlockV9(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_cbam=True,
                 is_lastblock=False):
        super(DecoderBlockV9, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam
        self.is_lastblock = is_lastblock

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        expanded_n_feats = n_feats * 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, expanded_n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(expanded_n_feats),
            nn.ReLU(inplace=True),
        )

        self.conv1_ = nn.Sequential(
            nn.Conv2d(expanded_n_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        if use_cbam is True:
            self.cbam = CBAM(edge_feats)

        self.conv2 = nn.Sequential(
            nn.Conv2d(edge_feats, edge_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats // 2),
            nn.ReLU(inplace=True),
        )

        self.conv2_ = nn.Sequential(
            nn.Conv2d(edge_feats // 2, output_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feature, edge, target=None):
        if self.block_sn >= 2:
            rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1_(self.conv1(x))

        if self.use_cbam is True:
            x = self.cbam(x)

        x = x + edge

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.conv2_(self.conv2(x))

        return x


class DecoderV9(nn.Module):
    def __init__(self):
        super(DecoderV9, self).__init__()

        self.decoder_block_1 = DecoderBlockV9(1, 64, 64, output_feats=32)
        self.decoder_block_2 = DecoderBlockV9(2, 64, 64, output_feats=32, target_feats=32)
        self.decoder_block_3 = DecoderBlockV9(3, 32, 16, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[1], edges[-2], target=y)
        y = self.decoder_block_3(features[0], edges[-3], target=y)

        return y


class DecoderBlockV10(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_cbam=True,
                 dpr=[0.025, 0.050, 0.075, 0.10]):
        super(DecoderBlockV10, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        self.conv1 = nn.Sequential(
            # nn.Conv2d(n_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            FasterMutableBlock(dim=n_feats, out_dim=edge_feats, drop_path=dpr[0]),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        if use_cbam is True:
            self.cbam = CBAM(edge_feats)

        if block_sn <= 2:
            self.conv2 = nn.Sequential(
                # nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                FasterMutableBlock(dim=edge_feats, out_dim=output_feats, drop_path=dpr[1]),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                # nn.Conv2d(edge_feats, edge_feats//2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                FasterMutableBlock(dim=edge_feats, out_dim=edge_feats//2, drop_path=dpr[2]),
                nn.BatchNorm2d(edge_feats//2),
                nn.ReLU(inplace=True),
                # nn.Conv2d(edge_feats//2, output_feats, kernel_size=1, stride=1, padding=0),
                FasterMutableBlock(dim=edge_feats//2, out_dim=output_feats, drop_path=dpr[3]),
            )

    def forward(self, rgb_feature, edge, target=None):
        if self.block_sn >= 2:
            rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.use_cbam is True:
            x = self.cbam(x)

        x = x + edge

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.conv2(x)

        return x


class DecoderV10(nn.Module):
    def __init__(self):
        super(DecoderV10, self).__init__()

        drop_path_rate = 0.1
        # 每个block用到了4个FasterMutableBlock drop概率逐步提升
        dpr = [x.item() for x in torch.linspace(0, end=drop_path_rate, steps=4)]

        self.decoder_block_1 = DecoderBlockV10(1, 64, 64, output_feats=32, dpr=dpr)
        self.decoder_block_2 = DecoderBlockV10(2, 64, 64, output_feats=32, target_feats=32, dpr=dpr)
        self.decoder_block_3 = DecoderBlockV10(3, 32, 16, output_feats=1, target_feats=32, dpr=dpr)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[1], edges[-2], target=y)
        y = self.decoder_block_3(features[0], edges[-3], target=y)

        return y


class DDRNetEdge(nn.Module):
    def __init__(self, opts):
        super(DDRNetEdge, self).__init__()

        bs = getattr(opts, "common.bs", 8)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # NOTE: DualResNet_Backbone+EdgeNet Params(5.70M)
        self.rgb_feature_extractor = DualResNet_Backbone(opts, pretrained=True, features=64)

        # NOTE: DDRNet_LiamEdge_EdgeNetV1_DecoderV6CBAM_06-25
        #       DDRNet_LiamEdge_EdgeNetV1_DecoderV6CBAM_kitti_09-15
        self.edge_feature_extractor = EdgeNetV1()  # NOTE: Params(0.25M)
        self.decoder = DecoderV6()  # NOTE: Params(0.22M)

        # self.edge_feature_extractor = EdgeNetV1()  # NOTE: Params(0.25M)

        # self.edge_feature_extractor = EdgeNetKSV1()  # NOTE: Params(0.17M)
        # self.edge_feature_extractor = EdgeNetV1FM()  # NOTE: Params(0.06M)
        # self.edge_feature_extractor = EdgeNetV2()  # NOTE: Params(0.02M)
        # self.edge_feature_extractor = EdgeNetV3()  # NOTE: Params(0.25M)

        # self.decoder = DecoderV1(use_cbam=True)  # NOTE: Params(0.26M)
        # self.decoder = DecoderKS()
        # self.decoder = DecoderKSV2()
        # self.decoder = DecoderV3()
        # self.decoder = DecoderV4(use_cbam=True)
        # self.decoder = DecoderV1PConv(use_cbam=True)  # NOTE: DecoderV1PConv Params(0.32M)
        # self.decoder = DecoderV1_()  # NOTE: Params(0.26M)
        # self.decoder = DecoderV5()  # NOTE: Params(0.23M)

        # self.decoder = DecoderV6()  # NOTE: Params(0.22M)

        # self.decoder = DecoderV7()  # NOTE: Params(M)
        # self.decoder = DecoderV8()    # NOTE: Params(M)
        # self.decoder = DecoderV9()  # NOTE: Params(M)
        # self.decoder = DecoderV10()  # NOTE: Params(0.29M)

    def forward(self, x):

        # # shape: [32, 32, 3, 3] [output_features, input_features, kernel_size, kernel_size]
        # shared_kernel_1 = self.rgb_feature_extractor.state_dict()['layer1.1.conv2.weight']
        # # print("shared_kernel_1.shape:", shared_kernel_1.shape)
        # # shape: [64, 64, 3, 3]
        # shared_kernel_2 = self.rgb_feature_extractor.state_dict()['layer2.1.conv2.weight']
        # # print("shared_kernel_2.shape:", shared_kernel_2.shape)
        # # shape: [64, 128, 3, 3]
        # shared_kernel_3 = self.rgb_feature_extractor.state_dict()['final_layer.conv1.weight']
        # # print("shared_kernel_3.shape:", shared_kernel_3.shape)

        # NOTE: 提取RGB的context特征信息
        # t0 = time.time()
        # [B, 32, 120, 160]  !  layer1
        # [B, 64, 60, 80]    !  layer2
        # [B, 128, 30, 40]      layer3
        # [B, 256, 15, 20]      layer4
        # [B, 64, 60, 80]    !  final_layer
        features = self.rgb_feature_extractor(x)  # [b, 64, 60, 80]
        # print("features, max:{}, min:{}".format(features[-1].max(), features[-1].min()))
        # print(features[-1])
        # print("rgb_feature_extractor time:{:.4f}s".format(time.time() - t0))

        # NOTE: 提取RGB的edge
        # t0 = time.time()
        # x_edge = edge_extractor(x, 'sobel')
        # _, _, x_edge = edge_extractor_torch(x, 'sobel', device=self.device)
        x_edge = self.edge_extractor_torch(x, 'sobel', device=self.device)
        # print("edge_extractor time:{:.6f}s".format(time.time() - t0))

        # NOTE: 提取edge特征信息
        # t0 = time.time()
        # [B, 16, 240, 320]  !
        # [B, 64, 120, 160]  !
        # [B, 64, 60, 80]    !
        edges = self.edge_feature_extractor(x_edge)

        # edges = self.edge_feature_extractor(
        #     x_edge,
        #     shared_kernel_1=shared_kernel_1,
        #     shared_kernel_2=shared_kernel_2,
        #     shared_kernel_3=shared_kernel_3
        # )
        # print("edges, max:{}, min:{}".
        # format(np.max(edges[-1].detach().cpu().numpy()), np.min(edges[-1].detach().cpu().numpy())))
        # print(edges[-1])
        # print("edge_feature_extractor time:{:.6f}s".format(time.time() - t0))

        # NOTE: 上采样稠密深度估计
        # t0 = time.time()
        # y, before_add, after_add = self.decoder(features, edges, x)
        # y, temp1, temp2 = self.decoder(features, edges, x)

        y = self.decoder(features, edges, x)

        # print("decoder time:{:.4f}s".format(time.time() - t0))

        # NOTE: 可视化
        # x_ = x
        # x_ = x_ * 255
        # x_ = x_.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        # b, _, _, _ = x.size()
        # print("b:", b)
        # x_edge = x_edge.detach().cpu().numpy().astype(np.uint8)
        # x_edge = x_edge.transpose(0, 2, 3, 1)
        #
        # cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
        # cmap_type = cmap_type_list[1]
        # for i in range(b):
        #     plt.subplot(2, 4, i + 1)
        #     plt.imshow(x_[i])  #
        #     plt.subplot(2, 4, i + 5)
        #     plt.imshow(x_edge[i], cmap=cmap_type)
        # plt.show()

        return y  # temp1, temp2  # y, edges, before_add, after_add

    def edge_extractor_torch(self, x: Tensor, mode="sobel", device="cuda:0"):
        conv_rgb_core_sobel_horizontal = [
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1],
             [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0],
             [1, 2, 1], [0, 0, 0], [-1, -2, -1],
             [0, 0, 0], [0, 0, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [1, 2, 1], [0, 0, 0], [-1, -2, -1],
             ]]
        conv_rgb_core_sobel_vertical = [
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
             [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0],
             [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
             [0, 0, 0], [0, 0, 0], [0, 0, 0]
             ],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
             ]]

        conv_op_horizontal = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
        conv_op_vertical = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)

        sobel_kernel_horizontal = np.array(conv_rgb_core_sobel_horizontal, dtype='float32')
        sobel_kernel_horizontal = sobel_kernel_horizontal.reshape((3, 3, 3, 3))
        conv_op_horizontal.weight.data = torch.from_numpy(sobel_kernel_horizontal).to(device)

        sobel_kernel_vertical = np.array(conv_rgb_core_sobel_vertical, dtype='float32')
        sobel_kernel_vertical = sobel_kernel_vertical.reshape((3, 3, 3, 3))
        conv_op_vertical.weight.data = torch.from_numpy(sobel_kernel_vertical).to(device)

        sobel_x = conv_op_horizontal(x)
        sobel_y = conv_op_vertical(x)
        sobel_xy = torch.sqrt(torch.square(sobel_x) + torch.square(sobel_y))

        # sobel_x = normalize2img_tensor(sobel_x)
        # sobel_y = normalize2img_tensor(sobel_y)
        sobel_xy = normalize2img_tensor(sobel_xy)

        return sobel_xy  # sobel_x, sobel_y, sobel_xy




