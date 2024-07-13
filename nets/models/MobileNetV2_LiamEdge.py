import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.DDRNet_23_slim import DualResNet_Backbone
from model.modules import Guided_Upsampling_Block
from ..modules.cbam import CBAM
from ..modules.se import SELayer
from ..modules.eca_module import eca_layer
from ..modules.srm_module import SRMLayer
from ..modules.gct_module import GCTLayer
from ..modules.GLPDepth_decoder import SelectiveFeatureFusion
from ..modules import InvertedResidual, BaseUpsamplingBlock
from ..layers import ConvLayer, LinearLayer, GlobalPool, Dropout
from ..modules.tf_block_topformer import Attention, BasicLayer, SemanticInjectionModule, get_shape, \
                                         CrossModalBasicLayer, CrossModalBasicLayerV2, CrossModalBasicLayerV1M1, \
                                         CrossModalBasicLayerV3, CrossModalBasicLayerV3M1, CrossModalBasicLayerV3M2, \
                                         CrossModalBasicLayerV3M3, CrossModalBasicLayerV3M4, CrossModalBasicLayerV3M5, \
                                         CrossModalBasicLayerV3M6

from ..modules.tf_block_TST import ConnectionModule
from ..modules.tf_block_SideRT import CrossScaleAttention, MultiScaleRefinement

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Union, Any

from utils import logger
from utils.math_utils import make_divisible

import time


def get_configuration(opts):
    mobilenetv2_config = {
        "layer1": {
            "expansion_ratio": 1,
            "out_channels": 16,  # 16
            "num_blocks": 1,
            "stride": 1,
        },
        "layer2": {
            "expansion_ratio": 6,
            "out_channels": 24,
            "num_blocks": 2,
            "stride": 2,
        },
        "layer3": {
            "expansion_ratio": 6,
            "out_channels": 32,
            "num_blocks": 3,
            "stride": 2,
        },
        "layer4": {
            "expansion_ratio": 6,
            "out_channels": 64,
            "num_blocks": 4,
            "stride": 2,
        },
        "layer4_a": {
            "expansion_ratio": 6,
            "out_channels": 96,
            "num_blocks": 3,
            "stride": 1,
        },
        "layer5": {
            "expansion_ratio": 6,
            "out_channels": 160,
            "num_blocks": 3,
            "stride": 2,
        },
        "layer5_a": {
            "expansion_ratio": 6,
            "out_channels": 320,
            "num_blocks": 1,
            "stride": 1,
        },
    }
    return mobilenetv2_config


# NOTE: 原来版本
def normalize2img_tensor(x: Tensor):
    min_val = x.min()
    max_val = x.max()
    res = (x - min_val) / (max_val - min_val)
    res = res * 255.
    return res


# def normalize2img_tensor(x: Tensor):
#     # temp = torch.min(x)
#     # print('type(temp):{}'.format(type(temp)))
#     min_val = torch.min(x).item()
#     # print('type(min_val):{}'.format(type(min_val)))
#     max_val = torch.max(x).item()
#     res = (x - min_val) / (max_val - min_val)
#     res = res * 255.
#     return res


# def normalize2img_tensor(x: Tensor):
#     min_val = torch.min(x)
#     max_val = torch.max(x)
#     res = (x - min_val) / (max_val - min_val)
#     res = res * 255.
#     return res


def normalize(x: np.ndarray):
    min_val = x.min()
    max_val = x.max()
    return (x - min_val) / (max_val - min_val)


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
            canny_edge = normalize(canny_edge)  # 将数据缩放到[0, 1]区间
            edge_batch_tensor[i, :, :, :] = torch.from_numpy(canny_edge).type(torch.float32)
        edge = edge_batch_tensor.to(device)  # [b, 1, h, w]
    elif mode == 'laplacian':
        # NOTE: Laplacian TODO
        pass

    return edge


class MVEdgeNetV1(nn.Module):
    def __init__(self):
        super(MVEdgeNetV1, self).__init__()

        # Params: 0.07M
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # kernel: [8, 3, 3, 3]
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # kernel: [16, 8, 3, 3]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(16, 20, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 24, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(24, 28, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(28),
            nn.Conv2d(28, 32, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # self.squeeze_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 24, 120, 160]
        features.append(x)
        x = self.edge_encoder_3(x)  # [B, 32, 60, 80]
        features.append(x)
        x = self.edge_encoder_4(x)  # [B, 64, 30, 40]
        features.append(x)
        return features


class MVEdgeNetV2(nn.Module):
    def __init__(self):
        super(MVEdgeNetV2, self).__init__()

        # Params: 0.07M
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # kernel: [8, 3, 3, 3]
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # kernel: [16, 8, 3, 3]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(16, 20, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 24, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(24, 28, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(28),
            nn.Conv2d(28, 32, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_5 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        # self.squeeze_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 24, 120, 160]
        features.append(x)
        x = self.edge_encoder_3(x)  # [B, 32, 60, 80]
        features.append(x)
        x = self.edge_encoder_4(x)  # [B, 64, 30, 40]
        features.append(x)
        x = self.edge_encoder_5(x)  # [B, 96, 30, 40]
        features.append(x)
        return features


class EdgeNetV3(nn.Module):
    def __init__(self, reduction=1):
        super(EdgeNetV3, self).__init__()

        intermediate_feats = 16 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, transition_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 16, h/4, w/4]

        conv_output_channels = intermediate_feats
        intermediate_feats = 32 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 32, h/8, w/8]

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 64, h/16, w/16]

        conv_output_channels = intermediate_feats
        intermediate_feats = 128 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 128, h/32, w/32]

    def forward(self, x):
        features = []

        x = self.edge_encoder_1(x)
        features.append(x)
        x = self.edge_encoder_2(x)
        features.append(x)
        x = self.edge_encoder_3(x)
        features.append(x)
        x = self.edge_encoder_4(x)
        features.append(x)

        return features


class EdgeNetV3M1(nn.Module):
    def __init__(self, reduction=1):
        super(EdgeNetV3M1, self).__init__()

        intermediate_feats = 32 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, transition_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 32, h/4, w/4]

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 64, h/8, w/8]

        conv_output_channels = intermediate_feats
        intermediate_feats = 128 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 128, h/16, w/16]

        conv_output_channels = intermediate_feats
        intermediate_feats = 160 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 160, h/32, w/32]

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)
        features.append(x)
        x = self.edge_encoder_2(x)
        features.append(x)
        x = self.edge_encoder_3(x)
        features.append(x)
        x = self.edge_encoder_4(x)
        features.append(x)

        return features


class EdgeHeadV1(nn.Module):
    def __init__(self, in_c):
        """
        Args:
            in_c: default 384
        """
        super(EdgeHeadV1, self).__init__()
        self.squeeze_conv = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU6()  # nn.Sigmoid()
        )

    def forward(self, edge_features):
        x1 = edge_features[0]  # 1/4
        x2 = F.interpolate(edge_features[1], scale_factor=2.0, mode='bilinear')  # 1/8 -> 1/4
        x3 = F.interpolate(edge_features[2], scale_factor=4.0, mode='bilinear')  # 1/16 -> 1/4
        x4 = F.interpolate(edge_features[3], scale_factor=8.0, mode='bilinear')  # 1/32 -> 1/4

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.squeeze_conv(x)  # 384 -> 32

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')  # 1/4 -> 1/2
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')  # 1/2 -> 1/1
        x = self.conv2(x)

        return x


class EdgeNetV4(nn.Module):
    def __init__(self, reduction=1):
        super(EdgeNetV4, self).__init__()

        intermediate_feats = 16 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, transition_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 16, h/2, w/2]

        conv_output_channels = intermediate_feats
        intermediate_feats = 32 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 32, h/4, w/4]

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 64, h/8, w/8]

        self.squeeze_conv = nn.Conv2d(intermediate_feats, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = []

        x = self.edge_encoder_1(x)
        x = self.edge_encoder_2(x)
        x = self.edge_encoder_3(x)
        x = self.squeeze_conv(x)  # [b, 1, h/8, w/8]
        x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear')  # [b, 1, h/16, w/16]
        x2 = F.interpolate(x, scale_factor=0.25, mode='bilinear')  # [b, 1, h/32, w/32]

        f0 = x.repeat(1, 64, 1, 1)  # [b, 64, h/8, w/8]
        f1 = x1.repeat(1, 128, 1, 1)  # [b, 128, h/16, w/16]
        f2 = x2.repeat(1, 160, 1, 1)  # [b, 160, h/32, w/32]

        features.append(f2)
        features.append(f1)
        features.append(f0)

        return features


class EdgeNetV4M1(nn.Module):
    def __init__(self, reduction=1):
        super(EdgeNetV4M1, self).__init__()

        intermediate_feats = 16 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, transition_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 16, h/2, w/2]

        conv_output_channels = intermediate_feats
        intermediate_feats = 32 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 32, h/4, w/4]

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.ReLU6(inplace=True),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )  # [b, 64, h/8, w/8]

        self.squeeze_conv = nn.Conv2d(intermediate_feats, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = []

        x = self.edge_encoder_1(x)
        x = self.edge_encoder_2(x)  # [b, 32, h/4, w/4]
        shortcut = x
        x = self.edge_encoder_3(x)
        x = self.squeeze_conv(x)  # [b, 1, h/8, w/8]
        x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear')  # [b, 1, h/16, w/16]
        x2 = F.interpolate(x, scale_factor=0.25, mode='bilinear')  # [b, 1, h/32, w/32]

        f0 = x.repeat(1, 64, 1, 1)  # [b, 64, h/8, w/8]
        f1 = x1.repeat(1, 128, 1, 1)  # [b, 128, h/16, w/16]
        f2 = x2.repeat(1, 160, 1, 1)  # [b, 160, h/32, w/32]

        features.append(f2)
        features.append(f1)
        features.append(f0)
        features.append(shortcut)

        return features


class EdgePyramid(nn.Module):
    def __init__(self):
        super(EdgePyramid, self).__init__()
        self.squeeze_conv = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, edge):
        features = []

        x = self.squeeze_conv(edge)  # [b, 1, h, w]
        x1 = F.interpolate(x, scale_factor=0.125, mode='bilinear')  # [b, 1, h/8, w/8]
        x2 = F.interpolate(x, scale_factor=0.0625, mode='bilinear')  # [b, 1, h/16, w/16]
        x3 = F.interpolate(x, scale_factor=0.03125, mode='bilinear')  # [b, 1, h/32, w/32]

        f1 = x1.repeat(1, 64, 1, 1)  # [b, 64, h/8, w/8]
        f2 = x2.repeat(1, 128, 1, 1)  # [b, 128, h/16, w/16]
        f3 = x3.repeat(1, 160, 1, 1)  # [b, 160, h/32, w/32]

        features.append(f3)
        features.append(f2)
        features.append(f1)

        return features


class EdgePyramidV2(nn.Module):
    def __init__(self):
        super(EdgePyramidV2, self).__init__()
        # self.squeeze_conv = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        pass

    def forward(self, edge):
        features = []

        r = edge[:, 0:1, :, :]
        g = edge[:, 1:2, :, :]
        b = edge[:, 2:3, :, :]

        x = 0.3 * r + 0.59 * g + 0.11 * b  # [b, 1, h, w]
        # print('edge.shape:{}'.format(edge.shape))

        x1 = F.interpolate(x, scale_factor=0.125, mode='bilinear')  # [b, 1, h/8, w/8]
        x2 = F.interpolate(x, scale_factor=0.0625, mode='bilinear')  # [b, 1, h/16, w/16]
        x3 = F.interpolate(x, scale_factor=0.03125, mode='bilinear')  # [b, 1, h/32, w/32]

        f1 = x1.repeat(1, 64, 1, 1)  # [b, 64, h/8, w/8]
        f2 = x2.repeat(1, 128, 1, 1)  # [b, 128, h/16, w/16]
        f3 = x3.repeat(1, 160, 1, 1)  # [b, 160, h/32, w/32]

        features.append(f3)
        features.append(f2)
        features.append(f1)

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
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_cbam=True,
                 is_lastblock=False):
        super(DecoderBlockV1, self).__init__()

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
        self.conv2 = nn.Sequential(
            nn.Conv2d(rgb_feats, output_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feature, edge, target=None):
        if target is None:
            x = torch.cat((edge, rgb_feature), dim=1)
        else:
            x = torch.cat((edge, rgb_feature, target), dim=1)

        if self.use_cbam is True:
            x = self.cbam(x)

        x = self.conv1(x)
        x = self.SE_block(x)
        x = x + rgb_feature

        if self.is_lastblock is False:
            x = self.conv2(x)
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            x = self.conv2(x)

        return x


class DecoderV1(nn.Module):
    def __init__(self):
        super(DecoderV1, self).__init__()

        self.decoder_block_1 = DecoderBlockV1(96, 64, output_feats=64)
        self.decoder_block_2 = DecoderBlockV1(32, 32, output_feats=24, target_feats=64)
        self.decoder_block_3 = DecoderBlockV1(24, 24, output_feats=16, target_feats=24)
        self.decoder_block_4 = DecoderBlockV1(16, 16, output_feats=1, target_feats=16, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)
        y = self.decoder_block_4(features[-4], edges[-4], target=y)

        return y


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
        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)
        if self.use_cbam is True:
            x = self.cbam(x)
        x = x + edge

        if self.block_sn == 1:
            x = self.conv2(x)
            return x

        if self.is_lastblock is False:
            x = self.conv2(x)
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            return x
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            x = self.conv2(x)
            return x


class DecoderV5(nn.Module):
    def __init__(self):
        super(DecoderV5, self).__init__()

        self.decoder_block_1 = DecoderBlockV5(1, 128, 96, output_feats=96)
        self.decoder_block_2 = DecoderBlockV5(2, 96, 64, output_feats=64, target_feats=96)
        self.decoder_block_3 = DecoderBlockV5(3, 32, 32, output_feats=24, target_feats=64)
        self.decoder_block_4 = DecoderBlockV5(4, 24, 24, output_feats=16, target_feats=24)
        self.decoder_block_5 = DecoderBlockV5(5, 16, 16, output_feats=1, target_feats=16, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)
        y = self.decoder_block_4(features[-4], edges[-4], target=y)
        y = self.decoder_block_5(features[-5], edges[-5], target=y)

        return y


class DecoderBlockV6(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 scale_factor=2.0,
                 attn_type='cbam',
                 is_lastblock=False):
        super(DecoderBlockV6, self).__init__()
        self.block_sn = block_sn
        self.scale_factor = scale_factor
        self.attn_type = attn_type
        self.is_lastblock = is_lastblock

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU6(inplace=True),
        )

        if attn_type == 'cbam':
            self.attn = CBAM(edge_feats)
        elif attn_type == 'cbamc':  # 只包含Channel通道注意力
            self.attn = CBAM(edge_feats, no_spatial=True)
        elif attn_type == 'cbams':  # 只包含Spatial空间注意力
            self.attn = CBAM(edge_feats, no_channel=True)
        elif attn_type == 'se':
            self.attn = SELayer(edge_feats)
        elif attn_type == 'eca':
            self.attn = eca_layer(edge_feats)
        elif attn_type == 'srm':
            self.attn = SRMLayer(edge_feats)
        elif attn_type == 'gct':
            self.attn = GCTLayer(edge_feats)
        else:
            logger.error('{} attention mechanism is not supported yet.'.format(attn_type))
            exit()

        if is_lastblock is False:
            if scale_factor == 2.0:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(output_feats),
                    nn.ReLU6(inplace=True),
                )
            elif scale_factor == 4.0:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(edge_feats, edge_feats, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(edge_feats),
                    nn.ReLU6(inplace=True),
                )
                self.conv3 = nn.Sequential(
                    nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(output_feats),
                    nn.ReLU6(inplace=True),
                )
        else:
            if scale_factor == 2.0:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(edge_feats, edge_feats//2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                    nn.BatchNorm2d(edge_feats//2),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(edge_feats//2, 1, kernel_size=1, stride=1, padding=0),
                )
            elif scale_factor == 4.0:
                self.conv2 = nn.Sequential(
                    nn.Conv2d(edge_feats, edge_feats, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(edge_feats),
                    nn.ReLU6(inplace=True),
                )
                self.conv3 = nn.Sequential(
                    nn.Conv2d(edge_feats, edge_feats//2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                    nn.BatchNorm2d(edge_feats//2),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(edge_feats//2, 1, kernel_size=1, stride=1, padding=0),
                )

    def forward(self, rgb_feature, edge, target=None):
        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.attn_type is not None:
            x = self.attn(x)

        x = x + edge

        if self.scale_factor == 2.0:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            x = self.conv2(x)
        elif self.scale_factor == 4.0:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            x = self.conv2(x)
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
            x = self.conv3(x)

        return x


class DecoderV6M1(nn.Module):
    def __init__(self, attn_type='cbam'):
        super(DecoderV6M1, self).__init__()

        self.decoder_block_1 = DecoderBlockV6(1, 320, 128, 128, scale_factor=2.0, attn_type=attn_type)
        self.decoder_block_2 = DecoderBlockV6(2, 96, 64, 64, 128, scale_factor=2.0, attn_type=attn_type)
        self.decoder_block_3 = DecoderBlockV6(3, 32, 32, 32, 64, scale_factor=2.0, attn_type=attn_type)
        self.decoder_block_4 = DecoderBlockV6(4, 24, 16, 1, 32, scale_factor=4.0, attn_type=attn_type,
                                              is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)
        y = self.decoder_block_4(features[-4], edges[-4], target=y)

        return y


class DecoderV6M2(nn.Module):
    def __init__(self, attn_type='cbam'):
        super(DecoderV6M2, self).__init__()

        self.decoder_block_1 = DecoderBlockV6(1, 160, 128, 128, scale_factor=2.0, attn_type=attn_type)
        self.decoder_block_2 = DecoderBlockV6(2, 128, 64, 64, 128, scale_factor=2.0, attn_type=attn_type)
        self.decoder_block_3 = DecoderBlockV6(3, 64, 32, 32, 64, scale_factor=2.0, attn_type=attn_type)
        self.decoder_block_4 = DecoderBlockV6(4, 32, 16, 1, 32, scale_factor=4.0, attn_type=attn_type,
                                              is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)
        y = self.decoder_block_4(features[-4], edges[-4], target=y)

        return y


class OutputHead(nn.Module):
    """
    NOTE: PPT中的 OutputHeadV1 OHV1
    """
    def __init__(self, in_c):
        super(OutputHead, self).__init__()

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_c),
            nn.ReLU6(inplace=True),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c // 2),
            nn.ReLU6(inplace=True),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(in_c // 2, 1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(in_c // 2),
            # nn.ReLU6(inplace=True),
        )

    def agg_res(self, preds):
        outs = preds[0]  # [b, 256, h/4, w/4]
        for pred in preds[1:]:
            pred = F.interpolate(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def forward(self, features):
        x = self.agg_res(features)
        x = self.linear_fuse(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.up_conv1(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.up_conv2(x)

        return x


class OutputHeadV1M1(nn.Module):
    def __init__(self, in_c):
        super(OutputHeadV1M1, self).__init__()

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_c),
            nn.ReLU6(inplace=True),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU6(inplace=True),
        )

        self.squeeze_conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_c // 2),
            nn.ReLU6(inplace=True),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c // 2),
            nn.ReLU6(inplace=True),
        )

        self.squeeze_conv2 = nn.Sequential(
            nn.Conv2d(in_c // 2, 1, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(1),
            # nn.ReLU6(inplace=True),
        )

    def agg_res(self, preds):
        outs = preds[0]  # [b, 256, h/4, w/4]
        for pred in preds[1:]:
            pred = F.interpolate(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def forward(self, features):
        x = self.agg_res(features)
        x = self.linear_fuse(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = x + self.up_conv1(x)  # residual
        x = self.squeeze_conv1(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = x + self.up_conv2(x)  # residual
        x = self.squeeze_conv2(x)

        return x


class OutputHeadV2(nn.Module):
    def __init__(self, rgb_feats, edge_feats):
        super(OutputHeadV2, self).__init__()

        self.linear_fuse_rgb = nn.Sequential(
            nn.Conv2d(rgb_feats, rgb_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(rgb_feats),
            nn.ReLU6(inplace=True),
        )

        self.linear_fuse_edge = nn.Sequential(
            nn.Conv2d(edge_feats, edge_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU6(inplace=True),
        )

        intermediate_feats = (rgb_feats + edge_feats) // 2

        self.concat_conv = nn.Sequential(
            nn.Conv2d(rgb_feats + edge_feats, intermediate_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(intermediate_feats, intermediate_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_feats // 2),
            nn.ReLU6(inplace=True),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(intermediate_feats // 2, 1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(in_c // 2),
            # nn.ReLU6(inplace=True),
        )

    def agg_res(self, preds):
        outs = preds[0]  # [b, 256, h/4, w/4]
        for pred in preds[1:]:
            pred = F.interpolate(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def forward(self, rgb_features, edge_features):
        rgb = self.agg_res(rgb_features)
        rgb = self.linear_fuse_rgb(rgb)

        edge = self.agg_res(edge_features)
        edge = self.linear_fuse_edge(edge)

        x = torch.cat((rgb, edge), dim=1)
        x = self.concat_conv(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.up_conv1(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.up_conv2(x)

        return x


class OutputHeadV3(nn.Module):
    def __init__(self, rgb_feats, edge_feats):
        super(OutputHeadV3, self).__init__()

        self.linear_fuse_rgb = nn.Sequential(
            nn.Conv2d(rgb_feats, rgb_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(rgb_feats),
            nn.ReLU6(inplace=True),
        )

        self.linear_fuse_edge = nn.Sequential(
            nn.Conv2d(edge_feats, edge_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU6(inplace=True),
        )

        intermediate_feats = edge_feats

        self.concat_conv = nn.Sequential(
            nn.Conv2d(rgb_feats + edge_feats, intermediate_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU6(inplace=True),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(intermediate_feats, intermediate_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_feats // 2),
            nn.ReLU6(inplace=True),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(intermediate_feats // 2, 1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(in_c // 2),
            # nn.ReLU6(inplace=True),
        )

    def agg_res(self, preds):
        outs = preds[0]  # [b, 256, h/4, w/4]
        for pred in preds[1:]:
            pred = F.interpolate(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def forward(self, rgb_features, edge_features):
        rgb = self.agg_res(rgb_features)
        rgb = self.linear_fuse_rgb(rgb)

        edge = self.agg_res(edge_features)
        edge = self.linear_fuse_edge(edge)
        shortcut = edge

        x = torch.cat((rgb, edge), dim=1)
        x = self.concat_conv(x)
        x = x + shortcut
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.up_conv1(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.up_conv2(x)

        return x


class OutputHeadV4(nn.Module):
    def __init__(self,
                 in_channels=[160, 128, 64],
                 out_channels=64):
        super(OutputHeadV4, self).__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)  # 32 out_channels

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        max_res_fm_channels = 32  # 最大分辨率的特征图的通道数
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)  # out_channels
        self.fusion3 = SelectiveFeatureFusion(max_res_fm_channels)  # out_channels
        self.squeeze_conv = nn.Conv2d(out_channels, max_res_fm_channels, kernel_size=1, stride=1, padding=0)

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(max_res_fm_channels, max_res_fm_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max_res_fm_channels, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, features):
        # for i in range(len(features)):
        #     print('features[{}].shape: {}'.format(i, features[i].shape))

        x_4_ = self.bot_conv(features[0])
        out = self.up(x_4_)

        x_3_ = self.skip_conv1(features[1])
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(features[2])
        out = self.fusion2(x_2_, out)
        out = self.up(out)
        out = self.squeeze_conv(out)
        # print('out.shape:{}'.format(out.shape))

        out = self.fusion3(features[3], out)
        out = self.up(out)
        out = self.up(out)

        out = self.last_layer_depth(out)

        return out


class OutputHeadV5(nn.Module):
    def __init__(self,
                 in_channels=[160, 128],
                 out_channels=64,
                 max_depth=10.):
        super(OutputHeadV5, self).__init__()
        self.max_depth = max_depth

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)

        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU6(inplace=False),
            nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, features):
        x0 = self.bot_conv(features[-1])    # [b, 160, h/32, w/32] -> [b, 64, h/32, w/32]
        x1 = self.skip_conv1(features[-2])  # [b, 160, h/32, w/32] -> [b, 64, h/32, w/32]

        y1 = self.fusion1(x1, x0)  # [b, 64, h/32, w/32]
        y1 = F.interpolate(y1, scale_factor=2.0, mode='bilinear')  # [b, 64, h/16, w/16]
        # print('y1.shape:{}'.format(y1.shape))

        x2 = self.skip_conv2(features[-3])  # [b, 128, h/16, w/16] -> [b, 64, h/16, w/16]

        y2 = self.fusion2(x2, y1)  # [b, 64, h/16, w/16]
        y2 = F.interpolate(y2, scale_factor=2.0, mode='bilinear')  # [b, 64, h/8, w/8]
        # print('y2.shape:{}'.format(y2.shape))

        y3 = self.fusion3(features[0], y2)  # [b, 64, h/8, w/8]
        y3 = F.interpolate(y3, scale_factor=2.0, mode='bilinear')  # [b, 64, h/4, w/4]
        # print('y3.shape:{}'.format(y3.shape))

        y = self.conv1(y3)
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')  # [b, 64, h/2, w/2]
        y = self.conv2(y)
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')  # [b, 64, h/1, w/1]

        y = self.last_layer_depth(y)
        y = torch.sigmoid(y) * self.max_depth  # NOTE: 原GLPDepth代码 MobileNetV2-TP_TST_OHV5_08-10
        # y = 1 / torch.sigmoid(y)  # NOTE: GuidedDepth中的网络输出的是DepthNorm形式 MobileNetV2-TP_TST_OHV5_DepthNorm_08-10

        return y


class OutputHeadV6(nn.Module):
    def __init__(self):
        super(OutputHeadV6, self).__init__()
        self.msr1 = MultiScaleRefinement(64, None, out_dim=32)
        self.msr2 = MultiScaleRefinement(32, None, out_dim=16)
        self.msr3 = MultiScaleRefinement(16, None, out_dim=8)
        self.msr4 = MultiScaleRefinement(8, None, out_dim=1)

    def forward(self, features):
        y1 = self.msr1(features[1], features[0])
        y2 = self.msr2(features[2], y1)
        y3 = self.msr3(y2)
        y4 = self.msr4(y3)
        return y4


class OutputHeadV7(nn.Module):
    def __init__(self, max_depth):
        super(OutputHeadV7, self).__init__()
        self.max_depth = max_depth

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(160 + 160, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU6()
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6()
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(128 + 128 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6()
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6()
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64 + 64 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6()
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.out_conv1 = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6()
        )

        # self.out_conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()  # Sigmoid会出现负值深度
            # nn.BatchNorm2d(16),
            # nn.ReLU6()
        )

    def forward(self, fused, edges):
        # for i in range(len(fused)):
        #     print('fused[{}].shape:{}'.format(i, fused[i].shape))
        # for i in range(len(edges)):
        #     print('edges[{}].shape:{}'.format(i, edges[i].shape))

        x1 = torch.cat((fused[0], edges[0]), dim=1)
        x1 = self.conv1_1(x1)
        x1 = x1 + edges[0]
        x1 = F.interpolate(x1, scale_factor=2.0, mode='bilinear')  # 1/16
        x1 = self.conv1_2(x1)

        x2 = torch.cat((x1, fused[1], edges[1]), dim=1)
        x2 = self.conv2_1(x2)
        x2 = x2 + edges[1]
        x2 = F.interpolate(x2, scale_factor=2.0, mode='bilinear')  # 1/8
        x2 = self.conv2_2(x2)

        x3 = torch.cat((x2, fused[2], edges[2]), dim=1)
        x3 = self.conv3_1(x3)
        x3 = x3 + edges[2]
        x3 = F.interpolate(x3, scale_factor=2.0, mode='bilinear')  # 1/4
        x3 = self.conv3_2(x3)

        x4 = torch.cat((x3, fused[3], edges[3]), dim=1)
        x4 = F.interpolate(x4, scale_factor=2.0, mode='bilinear')  # 1/2
        x4 = self.out_conv1(x4)

        x5 = F.interpolate(x4, scale_factor=2.0, mode='bilinear')  # 1/1
        x5 = self.out_conv2(x5)
        # y = torch.sigmoid(x5) * self.max_depth
        # y = torch.sigmoid(x5)

        return x5


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])  # encoder最后一层的输出形状[b, c, h/32, w/32]
        H = (H - 1) // self.stride + 1  # 进一步下采样h/64
        W = (W - 1) // self.stride + 1  # 进一步下采样w/64
        # 输出的形状仍然是[b, c, h/64, w/64] encoder四层的输出在通道维度上cat 分辨率都是原始的1/64
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class BackboneMultiScaleAttnExtractor3(nn.Module):
    """
    Backbone Multi Scale Attention Extractor (BMSAE3)
    """
    def __init__(self,
                 img_shape=(480, 640),
                 feats_list=[32, 64, 128, 160],
                 out_channels=[256, 256, 256, 256]):
        super(BackboneMultiScaleAttnExtractor3, self).__init__()
        h, w = img_shape[0], img_shape[1]
        self.feats_list = feats_list

        self.ppa = PyramidPoolAgg(stride=2)

        depths = 4
        drop_path_rate = 0.1
        embed_dim = sum(feats_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        self.trans = BasicLayer(
            block_num=depths,  # 4
            embedding_dim=embed_dim,  # 384
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,  # [0.025, 0.050, 0.075, 0.100]
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer)  # nn.ReLU6

        self.sim = nn.ModuleList()
        for i in range(len(feats_list)):
            self.sim.append(
                SemanticInjectionModule(feats_list[i], out_channels[i], norm_cfg=norm_cfg, activations=None)
            )

    def forward(self, features):
        x = self.ppa(features)
        x = self.trans(x)

        out_features = []
        global_tokens = x.split(self.feats_list, dim=1)
        # for i in range(len(global_tokens)):
        #     print('global_tokens[{}].shape:{}'.format(i, global_tokens[i].shape))

        for i in range(len(self.feats_list)):
            out_features.append(self.sim[i](features[i], global_tokens[i]))
            # print('out_features[{}].shape:{}'.format(i, out_features[i].shape))

        # print('x.shape:{}'.format(x.shape))
        return out_features


class TransitionModuleV1(nn.Module):
    def __init__(self,
                 feats_list=[32, 64, 128, 160],
                 out_channels=[256, 256, 256, 256],
                 alpha=0.5):
        super(TransitionModuleV1, self).__init__()
        self.feats_list = feats_list
        depths = 4
        drop_path_rate = 0.1
        embed_dim = sum(feats_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayer(
            block_num=depths,  # 4
            embedding_dim=embed_dim,  # 384
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,  # [0.025, 0.050, 0.075, 0.100]
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

        self.sim = nn.ModuleList()
        for i in range(len(feats_list)):
            self.sim.append(
                SemanticInjectionModule(feats_list[i], out_channels[i], norm_cfg=norm_cfg, activations=None)
            )

    def forward(self, rgb_feats, edge_feats):
        x = self.trans(rgb_feats, edge_feats)

        out_features = []
        global_tokens = x.split(self.feats_list, dim=1)
        # for i in range(len(global_tokens)):
        #     print('global_tokens[{}].shape:{}'.format(i, global_tokens[i].shape))

        for i in range(len(self.feats_list)):
            out_features.append(self.sim[i](rgb_feats[i], global_tokens[i]))

        return out_features  # x


class TransitionModuleV1M1(nn.Module):
    def __init__(self,
                 feats_list=[32, 64, 128, 160],
                 out_channels=[256, 256, 256, 256],
                 alpha=0.5):
        super(TransitionModuleV1M1, self).__init__()
        self.feats_list = feats_list
        depths = 4
        drop_path_rate = 0.1
        embed_dim = sum(feats_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayerV1M1(
            block_num=depths,  # 4
            embedding_dim=embed_dim,  # 384
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,  # [0.025, 0.050, 0.075, 0.100]
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

        self.sim = nn.ModuleList()
        for i in range(len(feats_list)):
            self.sim.append(
                SemanticInjectionModule(feats_list[i], out_channels[i], norm_cfg=norm_cfg, activations=None)
            )

    def forward(self, rgb_feats, edge_feats):
        x = self.trans(rgb_feats, edge_feats)

        out_features = []
        global_tokens = x.split(self.feats_list, dim=1)
        # for i in range(len(global_tokens)):
        #     print('global_tokens[{}].shape:{}'.format(i, global_tokens[i].shape))

        for i in range(len(self.feats_list)):
            out_features.append(self.sim[i](rgb_feats[i], global_tokens[i]))

        return out_features  # x


class TransitionModuleV2(nn.Module):
    def __init__(self,
                 feats_list=[32, 64, 128, 160],
                 out_channels=[256, 256, 256, 256],
                 alpha=0.5):
        super(TransitionModuleV2, self).__init__()
        self.feats_list = feats_list
        depths = 4
        drop_path_rate = 0.1
        embed_dim = sum(feats_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayerV2(
            block_num=depths,  # 4
            embedding_dim=embed_dim,  # 384
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,  # [0.025, 0.050, 0.075, 0.100]
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

        self.sim = nn.ModuleList()
        for i in range(len(feats_list)):
            self.sim.append(
                SemanticInjectionModule(feats_list[i], out_channels[i], norm_cfg=norm_cfg, activations=None)
            )

    def forward(self, rgb_feats, edge_feats):
        x = self.trans(rgb_feats, edge_feats)

        out_features = []
        global_tokens = x.split(self.feats_list, dim=1)
        # for i in range(len(global_tokens)):
        #     print('global_tokens[{}].shape:{}'.format(i, global_tokens[i].shape))

        for i in range(len(self.feats_list)):
            out_features.append(self.sim[i](rgb_feats[i], global_tokens[i]))

        return out_features  # x


class TransitionModuleV3(nn.Module):
    def __init__(self,
                 feats_list=[64, 128, 160],
                 alpha=0.5):
        super(TransitionModuleV3, self).__init__()
        depths = 3
        drop_path_rate = 0.1
        embed_dim_list = feats_list
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayerV3(
            block_num=depths,  # 3
            embedding_dim_list=embed_dim_list,  # [64, 128, 160]
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

    def forward(self, rgb_feats, edge_feats):
        features = self.trans(rgb_feats, edge_feats)
        return features


# NOTE: 原版
class TransitionModuleV3M1(nn.Module):
    def __init__(self,
                 feats_list=[64, 128, 160],
                 alpha=0.5):
        super(TransitionModuleV3M1, self).__init__()
        depths = 4
        drop_path_rate = 0.1
        embed_dim_list = feats_list
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayerV3M1(
            block_num=3,  # 3
            block_per_block_nums=depths,
            embedding_dim_list=embed_dim_list,  # [64, 128, 160]
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

    def forward(self, rgb_feats, edge_feats):
        features = self.trans(rgb_feats, edge_feats)
        return features


# NOTE: 可视化注意力图
# class TransitionModuleV3M1(nn.Module):
#     def __init__(self,
#                  feats_list=[64, 128, 160],
#                  alpha=0.5):
#         super(TransitionModuleV3M1, self).__init__()
#         depths = 4
#         drop_path_rate = 0.1
#         embed_dim_list = feats_list
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
#         norm_cfg = dict(type='BN2d', requires_grad=True)
#         act_layer = nn.ReLU6
#         # NOTE: CrossModalBasicLayer中包含了ppa操作
#         self.trans = CrossModalBasicLayerV3M1(
#             block_num=3,  # 3
#             block_per_block_nums=depths,
#             embedding_dim_list=embed_dim_list,  # [64, 128, 160]
#             key_dim=16,  # 16
#             num_heads=8,  # 8
#             mlp_ratio=2,  # 2
#             attn_ratio=2,  # 2
#             drop=0, attn_drop=0,
#             drop_path=dpr,
#             norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
#             act_layer=act_layer,  # nn.ReLU6
#             alpha=alpha)
#
#     def forward(self, rgb_feats, edge_feats):
#         features, attn_score_list = self.trans(rgb_feats, edge_feats)
#         return features, attn_score_list


class TransitionModuleV3M2(nn.Module):
    def __init__(self,
                 feats_list=[64, 128, 160],
                 alpha=0.5):
        super(TransitionModuleV3M2, self).__init__()
        depths = 3
        drop_path_rate = 0.1
        embed_dim_list = feats_list
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayerV3M2(
            block_num=depths,  # 3
            embedding_dim_list=embed_dim_list,  # [64, 128, 160]
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

    def forward(self, rgb_feats, edge_feats):
        features = self.trans(rgb_feats, edge_feats)
        return features


class TransitionModuleV3M3(nn.Module):
    def __init__(self,
                 feats_list=[64, 128, 160],
                 alpha=0.5):
        super(TransitionModuleV3M3, self).__init__()
        depths = 5  # 1个交叉TF块和5-1=4个基础TF块
        drop_path_rate = 0.1
        embed_dim_list = feats_list
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayerV3M3(
            block_num=3,  # 3
            block_per_block_nums=depths,
            embedding_dim_list=embed_dim_list,  # [64, 128, 160]
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

    def forward(self, rgb_feats, edge_feats):
        features = self.trans(rgb_feats, edge_feats)
        return features


class TransitionModuleV3M4(nn.Module):
    def __init__(self,
                 feats_list=[64, 128, 160],
                 alpha='learnable'):  # NOTE: alpha是可以学习的参数
        super(TransitionModuleV3M4, self).__init__()
        depths = 4
        drop_path_rate = 0.1
        embed_dim_list = feats_list
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayerV3M4(
            block_num=3,  # 3
            block_per_block_nums=depths,
            embedding_dim_list=embed_dim_list,  # [64, 128, 160]
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

    def forward(self, rgb_feats, edge_feats):
        features = self.trans(rgb_feats, edge_feats)
        return features


class TransitionModuleV3M5(nn.Module):
    def __init__(self,
                 feats_list=[64, 128, 160],
                 alpha=0.5):
        super(TransitionModuleV3M5, self).__init__()
        depths = 4
        drop_path_rate = 0.1
        embed_dim_list = feats_list
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayerV3M5(
            block_num=3,  # 3
            block_per_block_nums=depths,
            embedding_dim_list=embed_dim_list,  # [64, 128, 160]
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

    def forward(self, rgb_feats, edge_feats):
        features = self.trans(rgb_feats, edge_feats)
        return features


class TransitionModuleV3M6(nn.Module):
    def __init__(self,
                 feats_list=[64, 128, 160],
                 alpha='learnable'):
        super(TransitionModuleV3M6, self).__init__()
        depths = 4
        drop_path_rate = 0.1
        embed_dim_list = feats_list
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        # NOTE: CrossModalBasicLayer中包含了ppa操作
        self.trans = CrossModalBasicLayerV3M6(
            block_num=3,  # 3
            block_per_block_nums=depths,
            embedding_dim_list=embed_dim_list,  # [64, 128, 160]
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer,  # nn.ReLU6
            alpha=alpha)

    def forward(self, rgb_feats, edge_feats):
        features = self.trans(rgb_feats, edge_feats)
        return features


class TransitionModuleV4(nn.Module):
    def __init__(self):
        super(TransitionModuleV4, self).__init__()
        self.cas1 = CrossScaleAttention(128, 160, 64)
        self.cas2 = CrossScaleAttention(64, 64, 64)
        self.cas3 = CrossScaleAttention(32, 64, 32)

    def forward(self, features):
        x1 = self.cas1(features[1], features[0])
        x2 = self.cas2(features[2], x1)
        x3 = self.cas3(features[3], x2)
        return [x1, x2, x3]


class EdgeNetMultiScaleAttnExtractor1(nn.Module):
    """
    EdgeNet Multi-Scale Attention Extractor (EMSAE1)
    """
    def __init__(self,
                 img_shape=(480, 640),
                 feats_list=[16, 32, 64, 128],
                 out_channels=[160, 160, 160, 160]):
        super(EdgeNetMultiScaleAttnExtractor1, self).__init__()
        h, w = img_shape[0], img_shape[1]
        self.feats_list = feats_list

        self.ppa = PyramidPoolAgg(stride=2)

        depths = 4
        drop_path_rate = 0.1
        embed_dim = sum(feats_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        self.trans = BasicLayer(
            block_num=depths,  # 4
            embedding_dim=embed_dim,  # 240
            key_dim=16,  # 16
            num_heads=8,  # 8
            mlp_ratio=2,  # 2
            attn_ratio=2,  # 2
            drop=0, attn_drop=0,
            drop_path=dpr,  # [0.025, 0.050, 0.075, 0.100]
            norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
            act_layer=act_layer)  # nn.ReLU6

        self.sim = nn.ModuleList()
        for i in range(len(feats_list)):
            self.sim.append(
                SemanticInjectionModule(feats_list[i], out_channels[i], norm_cfg=norm_cfg, activations=None)
            )

    def forward(self, features):
        x = self.ppa(features)
        # print('x.shape:{}'.format(x.shape))
        x = self.trans(x)
        # print('x.shape:{}'.format(x.shape))

        out_features = []
        global_tokens = x.split(self.feats_list, dim=1)
        # for i in range(len(global_tokens)):
        #     print('global_tokens[{}].shape:{}'.format(i, global_tokens[i].shape))

        for i in range(len(self.feats_list)):
            out_features.append(self.sim[i](features[i], global_tokens[i]))
            # print('out_features[{}].shape:{}'.format(i, out_features[i].shape))

        return out_features


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))


class InvertedResidualTF(nn.Module):
    """
    NOTE: TopFormer
    """
    def __init__(
            self,
            inp: int,
            oup: int,
            ks: int,
            stride: int,
            expand_ratio: int,
            activations=None,
            norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InvertedResidualTF, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks // 2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TokenPyramidModule(nn.Module):
    def __init__(
            self,
            inp_channel=16,
            activation=nn.ReLU,
            norm_cfg=dict(type='BN', requires_grad=True),
            width_mult=1.):
        super().__init__()
        self.embed_out_indices = [2, 4, 6, 9]

        self.stem = nn.Sequential(
            Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),  # [b, 16, h/2, w/2]
            activation()
        )

        cfgs = [
            # k, t, c, s
            [3, 1, 16, 1],  # 1/2        0.464K  17.461M
            [3, 4, 32, 2],  # 1/4 1      3.44K   64.878M
            [3, 3, 32, 1],  # 4.44K   41.772M
            [5, 3, 64, 2],  # 1/8 3      6.776K  29.146M
            [5, 3, 64, 1],  # 13.16K  30.952M
            [3, 3, 128, 2],  # 1/16 5     16.12K  18.369M
            [3, 3, 128, 1],  # 41.68K  24.508M
            [5, 6, 160, 2],  # 1/32 7     0.129M  36.385M
            [5, 6, 160, 1],  # 0.335M  49.298M
            [3, 6, 160, 1],  # 0.335M  49.298M
        ]

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):  # topformer_base_1024x512_80k_2x8city.py中cfgs有十层
            output_channel = make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidualTF(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg,
                                       activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.embed_out_indices:
                outs.append(x)
        return outs


class SGNet(nn.Module):
    def __init__(self, out_feats=[160, 128, 64], intermediate_feat=96):
        super(SGNet, self).__init__()
        self.out_feats = out_feats
        # conv_module = nn.Sequential(
        #     nn.Conv2d(3, intermediate_feat, kernel_size=9, stride=1, padding=4),
        #     nn.BatchNorm2d(intermediate_feat),
        #     nn.ReLU6(inplace=True),
        #     nn.Conv2d(intermediate_feat, intermediate_feat // 2, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(intermediate_feat // 2),
        #     nn.ReLU6(inplace=True),
        #     nn.Conv2d(intermediate_feat // 2, 1, kernel_size=5, stride=1, padding=2)
        # )
        #
        # self.conv_module_list = nn.ModuleList()
        # for i in range(3):
        #     self.conv_module_list.append(conv_module)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, intermediate_feat, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(intermediate_feat),
            nn.ReLU6(inplace=True),
            nn.Conv2d(intermediate_feat, intermediate_feat // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(intermediate_feat // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(intermediate_feat // 2, 1, kernel_size=5, stride=1, padding=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, intermediate_feat, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(intermediate_feat),
            nn.ReLU6(inplace=True),
            nn.Conv2d(intermediate_feat, intermediate_feat // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(intermediate_feat // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(intermediate_feat // 2, 1, kernel_size=5, stride=1, padding=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(3, intermediate_feat, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(intermediate_feat),
            nn.ReLU6(inplace=True),
            nn.Conv2d(intermediate_feat, intermediate_feat // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(intermediate_feat // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(intermediate_feat // 2, 1, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        guidance_features = []
        x1 = F.interpolate(x, scale_factor=0.03125, mode='bilinear')  # 1/32
        x2 = F.interpolate(x, scale_factor=0.0625, mode='bilinear')  # 1/16
        x3 = F.interpolate(x, scale_factor=0.125, mode='bilinear')  # 1/8

        y1 = self.conv1(x1)
        y1 = y1.repeat(1, self.out_feats[0], 1, 1)
        y2 = self.conv2(x2)
        y2 = y2.repeat(1, self.out_feats[1], 1, 1)
        y3 = self.conv3(x3)
        y3 = y3.repeat(1, self.out_feats[2], 1, 1)

        guidance_features.append(y1)
        guidance_features.append(y2)
        guidance_features.append(y3)

        return guidance_features


max_depths = {
    'kitti': 80.0,
    'nyu_reduced': 10.0,
}


class MobileNetV2Edge(nn.Module):
    """
    This class defines the `MobileNetv2 architecture <https://arxiv.org/abs/1801.04381>`_
    """
    def __init__(self, opts, *args, **kwargs) -> None:
        super(MobileNetV2Edge, self).__init__()
        dataset_name = getattr(opts, "dataset.name", "nyu_reduced")  # opts原本为args 可能有误
        self.maxDepth = max_depths[dataset_name]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mode = getattr(opts, 'model.mode', '1.0')
        self.usage_mode = getattr(opts, 'common.mode', 'train')

        # NOTE: 完整版MobileNetV2 Params(1.81M)
        if self.mode == '1.0':
            self.dilation = 1
            self.round_nearest = 8
            self.dilate_l4 = False
            self.dilate_l5 = False

            width_mult = getattr(opts, "model.mobilenetv2.width_multiplier", 1.0)

            cfg = get_configuration(opts=opts)

            image_channels = 3
            input_channels = 32

            self.model_conf_dict = dict()

            # NOTE: Backbone Params(0.54M)
            self.conv_1 = ConvLayer(
                opts=opts,
                in_channels=image_channels,
                out_channels=input_channels,
                kernel_size=3,
                stride=2,
                use_norm=True,
                use_act=True,
            )
            self.model_conf_dict["conv1"] = {"in": image_channels, "out": input_channels}

            self.layer_1, out_channels = self._make_layer(
                opts=opts,
                mv2_config=cfg["layer1"],
                width_mult=width_mult,
                input_channel=input_channels,
            )
            self.model_conf_dict["layer1"] = {"in": input_channels, "out": out_channels}
            input_channels = out_channels

            self.layer_2, out_channels = self._make_layer(
                opts=opts,
                mv2_config=cfg["layer2"],
                width_mult=width_mult,
                input_channel=input_channels,
            )
            self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
            input_channels = out_channels

            self.layer_3, out_channels = self._make_layer(
                opts=opts,
                mv2_config=cfg["layer3"],
                width_mult=width_mult,
                input_channel=input_channels,
            )
            self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
            input_channels = out_channels

            self.layer_4, out_channels = self._make_layer(
                opts=opts,
                mv2_config=[cfg["layer4"], cfg["layer4_a"]],
                width_mult=width_mult,
                input_channel=input_channels,
                dilate=self.dilate_l4,
            )
            self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
            input_channels = out_channels

            self.layer_5, out_channels = self._make_layer(
                opts=opts,
                mv2_config=[cfg["layer5"], cfg["layer5_a"]],
                width_mult=width_mult,
                input_channel=input_channels,
                dilate=self.dilate_l5,
            )
            self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
            input_channels = out_channels

        # NOTE: TopFormer的Backbone Params(1.10M)
        elif self.mode == 'TokenPyramid':
            self.rgb_features_extractor = TokenPyramidModule()

        # self.extension_module = self._make_extension_module(opts=opts, input_channel=96)  # NOTE: Params(0.78M)

        # NOTE: sobel算子提取RGB图的边缘
        # conv_rgb_core_sobel_horizontal = [
        #     [[1, 2, 1], [0, 0, 0], [-1, -2, -1],
        #      [0, 0, 0], [0, 0, 0], [0, 0, 0],
        #      [0, 0, 0], [0, 0, 0], [0, 0, 0]
        #      ],
        #     [[0, 0, 0], [0, 0, 0], [0, 0, 0],
        #      [1, 2, 1], [0, 0, 0], [-1, -2, -1],
        #      [0, 0, 0], [0, 0, 0], [0, 0, 0]
        #      ],
        #     [[0, 0, 0], [0, 0, 0], [0, 0, 0],
        #      [0, 0, 0], [0, 0, 0], [0, 0, 0],
        #      [1, 2, 1], [0, 0, 0], [-1, -2, -1],
        #      ]]
        # conv_rgb_core_sobel_vertical = [
        #     [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
        #      [0, 0, 0], [0, 0, 0], [0, 0, 0],
        #      [0, 0, 0], [0, 0, 0], [0, 0, 0]
        #      ],
        #     [[0, 0, 0], [0, 0, 0], [0, 0, 0],
        #      [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
        #      [0, 0, 0], [0, 0, 0], [0, 0, 0]
        #      ],
        #     [[0, 0, 0], [0, 0, 0], [0, 0, 0],
        #      [0, 0, 0], [0, 0, 0], [0, 0, 0],
        #      [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
        #      ]]
        # self.conv_op_horizontal = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
        # self.conv_op_vertical = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
        # sobel_kernel_horizontal = np.array(conv_rgb_core_sobel_horizontal, dtype='float32')
        # sobel_kernel_horizontal = sobel_kernel_horizontal.reshape((3, 3, 3, 3))
        # self.conv_op_horizontal.weight.data = torch.from_numpy(sobel_kernel_horizontal).to(self.device)
        # sobel_kernel_vertical = np.array(conv_rgb_core_sobel_vertical, dtype='float32')
        # sobel_kernel_vertical = sobel_kernel_vertical.reshape((3, 3, 3, 3))
        # self.conv_op_vertical.weight.data = torch.from_numpy(sobel_kernel_vertical).to(self.device)
        # for param in self.conv_op_horizontal.parameters():
        #     param.requires_grad_(False)
        # for param in self.conv_op_vertical.parameters():
        #     param.requires_grad_(False)

        # NOTE: MobileNetV2-TP_TopFormer_OutputHead_07-30
        #       MobileNetV2-TP_TopFormer_OutputHead_09-14
        # self.rgb_feature_mhsa = BackboneMultiScaleAttnExtractor3()  # NOTE: Params(3.52M) 4.98 1.46
        # self.decoder = OutputHead(in_c=256)  # NOTE: Params(0.37M) 5.35 4.98

        # NOTE: MobileNetV2-TP_BMSAE3_ENV3_EMSAE1_OHV3_07-31
        #       MobileNetV2-TP_BMSAE3_ENV3_EMSAE1_OHV3_kitti_09-06
        # self.rgb_feature_mhsa = BackboneMultiScaleAttnExtractor3()  # NOTE: Params(3.52M) 4.98 1.46
        # self.edge_feature_extractor = EdgeNetV3(reduction=1)  # NOTE: Params(0.15M) 1.96 1.81
        # self.edge_feature_mhsa = EdgeNetMultiScaleAttnExtractor1()  # NOTE: Params(1.81M) 7.30 5.49
        # self.decoder = OutputHeadV3(rgb_feats=256, edge_feats=160)

        # NOTE: MobileNetV2-TP_ENV3M1_TMV1_OHV1_08-01
        # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        # self.transition_module = TransitionModuleV1(feats_list=[32, 64, 128, 160])
        # self.decoder = OutputHead(in_c=256)  # NOTE: Params(0.37M)

        # NOTE: MobileNetV2-TP_ENV3M1_TMV1_OHV1M1_08-02
        # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        # self.transition_module = TransitionModuleV1(feats_list=[32, 64, 128, 160])
        # self.decoder = OutputHeadV1M1(in_c=256)  # NOTE: Params(0.84M) 增加了很大的计算负担！！！

        # NOTE: MobileNetV2-TP_ENV3M1_TMV1_alpha1.0_OHV1_08-02
        # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        # self.transition_module = TransitionModuleV1(feats_list=[32, 64, 128, 160], alpha=1.0)
        # self.decoder = OutputHead(in_c=256)  # NOTE: Params(0.37M)

        # NOTE: MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_08-02
        #       MobileNetV2-TP_kitti_ENV3M1_TMV2_alpha0.5_OHV1_08-07
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_08-02
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_08-30
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_09-03
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_09-03
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_09-05
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_09-06
        # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        # self.transition_module = TransitionModuleV2(feats_list=[32, 64, 128, 160], alpha=0.5)
        # self.decoder = OutputHead(in_c=256)  # NOTE: Params(0.37M)

        # NOTE: MobileNetV2-TP_ENV3M1_EHV1_TMV2_OHV1_08-30
        #       MobileNetV2-TP_ENV3M1_EHV1_TMV2_OHV1_kitti_09-15
        # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        # self.transition_module = TransitionModuleV2(feats_list=[32, 64, 128, 160], alpha=0.5)
        # self.decoder = OutputHead(in_c=256)  # NOTE: Params(0.37M)
        # if self.usage_mode == 'train':
        #     self.edge_head = EdgeHeadV1(384)

        # NOTE: MobileNetV2-TP_ENV3M1_TMV1M1_alpha0.5_OHV1_08-02
        # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        # self.transition_module = TransitionModuleV1M1(feats_list=[32, 64, 128, 160], alpha=0.5)
        # self.decoder = OutputHead(in_c=256)  # NOTE: Params(0.37M)

        # NOTE: MobileNetV2-TP_ENV3M1_TMV3_alpha0.5_OHV4_08-03
        # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        # self.transition_module = TransitionModuleV3(feats_list=[64, 128, 160], alpha=0.5)
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_ENV3M1_TMV3M1_alpha0.5_OHV4_08-04
        # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        # self.transition_module = TransitionModuleV3M1(feats_list=[64, 128, 160], alpha=0.5)
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_ENV3M1_TMV3M2_alpha0.5_OHV4_08-05
        # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        # self.transition_module = TransitionModuleV3M2(feats_list=[64, 128, 160], alpha=0.5)
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_TST_OHV5_08-10
        #       MobileNetV2-TP_TST_OHV5_09-02
        # self.transition_module = ConnectionModule(block_num=3,
        #                                           local_dims=[64, 128, 160],
        #                                           global_dim=160,
        #                                           query_dims=[16, 16, 16],
        #                                           key_dims=[16, 16, 16],
        #                                           value_dims=[32, 32, 32],
        #                                           num_heads=[2, 4, 5],
        #                                           attn_ratio=2.0,
        #                                           mlp_ratio=2.0,
        #                                           drop=0.,
        #                                           attn_drop=0.,
        #                                           drop_path_rate=0.1,
        #                                           act_layer=nn.ReLU6,
        #                                           norm_cfg=dict(type='BN2d', requires_grad=True)
        #                                           )
        # self.decoder = OutputHeadV5(in_channels=[160, 128, 64], out_channels=64, max_depth=self.maxDepth)

        # NOTE: MobileNetV2-TP_SideRT_OHV6_08-11
        # self.transition_module = TransitionModuleV4()
        # self.decoder = OutputHeadV6()

        # NOTE: MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_08-26
        #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_09-03
        #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_kitti_09-04
        #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_kitti_09-04_2
        #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_09-04
        # self.edge_feature_extractor = EdgeNetV4(reduction=1)
        # self.transition_module = TransitionModuleV3M1(feats_list=[64, 128, 160], alpha=0.5)
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_EPV1_TMV3M1_alpha0.5_OHV4_08-27-
        # self.edge_feature_extractor = EdgePyramid()
        # self.transition_module = TransitionModuleV3M1(feats_list=[64, 128, 160], alpha=0.5)
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_EPV2_TMV3M1_alpha0.5_OHV4_08-29
        # self.edge_feature_extractor = EdgePyramidV2()
        # self.transition_module = TransitionModuleV3M1(feats_list=[64, 128, 160], alpha=0.5)
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_ENV4M1_TMV3M1_OHV7_09-03
        # self.edge_feature_extractor = EdgeNetV4M1(reduction=1)
        # self.transition_module = TransitionModuleV3M1(feats_list=[64, 128, 160], alpha=0.5)
        # self.decoder = OutputHeadV7(max_depth=self.maxDepth)

        # NOTE: MobileNetV2-TP_ENV4_TMV3M3_alpha0.5_OHV4_09-04
        # self.edge_feature_extractor = EdgeNetV4(reduction=1)
        # self.transition_module = TransitionModuleV3M3(feats_list=[64, 128, 160], alpha=0.5)
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_09-04
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti_09-05
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_09-26
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_09-27
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti_09-27
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_09-28
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-09
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-10
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-13
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-bts_10-16
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-16
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-17
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-bts_10-17
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-18
        self.edge_feature_extractor = EdgeNetV4(reduction=1)
        self.transition_module = TransitionModuleV3M4(feats_list=[64, 128, 160], alpha='learnable')
        self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_ENV4_TMV3M5_alpha0.5_OHV4_09-04
        # self.edge_feature_extractor = EdgeNetV4(reduction=1)
        # self.transition_module = TransitionModuleV3M5(feats_list=[64, 128, 160], alpha=0.5)
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_ENV4_TMV3M6_LearnableAlpha_OHV4_09-06
        # self.edge_feature_extractor = EdgeNetV4(reduction=1)
        # self.transition_module = TransitionModuleV3M6(feats_list=[64, 128, 160], alpha='learnable')
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_SGN_TMV3M4_LearnableAlpha_OHV4_09-26
        #       MobileNetV2-TP_SGN_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_09-28
        #       MobileNetV2-TP_SGN_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-09
        # self.guidance_feature_extractor = SGNet()  # NOTE: Params(0.09M)
        # self.transition_module = TransitionModuleV3M4(feats_list=[64, 128, 160], alpha='learnable')
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: MobileNetV2-TP_SGN_TMV3M4_ConstAlpha_OHV4_09-27
        # self.guidance_feature_extractor = SGNet()  # NOTE: Params(0.09M)
        # self.transition_module = TransitionModuleV3M4(feats_list=[64, 128, 160], alpha=[0.2, 0.4, 0.6])
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)




        # # self.edge_feature_extractor = MVEdgeNetV1()  # NOTE: Params(0.07M)
        # # self.edge_feature_extractor = MVEdgeNetV2()  # NOTE: Params(0.10M)
        # self.edge_feature_extractor = EdgeNetV3(reduction=1)  # NOTE: Params(0.15M) 1.96 1.81
        # # self.edge_feature_extractor = EdgeNetV3M1(reduction=1)
        #

        # self.rgb_feature_mhsa = BackboneMultiScaleAttnExtractor3()  # NOTE: Params(3.52M) 4.98 1.46

        # # self.decoder = DecoderV1()  # NOTE: Params(0.32M)
        # # self.decoder = DecoderV5()  # NOTE: Params(0.54M)
        # # self.decoder = DecoderV6M1()  # NOTE: Params(0.94M)  2.89  1.96
        # # self.decoder = DecoderV6M2()  # NOTE: Params(0.78M)  2.02  1.24
        # # self.decoder = GLPDecoder([320, 96, 32], 24, max_depth=10.0)  # NOTE: Params(0.06M) 1.87 1.81
        # self.decoder = OutputHead(in_c=256)  # NOTE: Params(0.37M) 5.35 4.98
        # self.decoder = OutputHeadV2(rgb_feats=256, edge_feats=160)  # NOTE: Params(1.07M) 8.01 6.94
        # # self.decoder = OutputHeadV3(rgb_feats=256, edge_feats=160)

    def rgb_feature_extractor(self, x: Tensor):
        features = None
        if self.mode == '1.0':
            features = []
            x = self.conv_1(x)
            x = self.layer_1(x)  # [B, 16, 240, 320]  !
            # features.append(x)
            x = self.layer_2(x)  # [B, 24, 120, 160]  !  ✔
            features.append(x)
            x = self.layer_3(x)  # [B, 32, 60, 80]    !  ✔
            features.append(x)
            x = self.layer_4(x)  # [B, 96, 30, 40]    !  ✔
            features.append(x)
            x = self.layer_5(x)  # [B, 320, 15, 20]      ✔
            features.append(x)
            # x = self.extension_module(x)  # [B, 96, 30, 40]    !
            # features.append(x)
        elif self.mode == 'TokenPyramid':
            # [B, 32, 120, 160]  1/4  ✔
            # [8, 64, 60, 80]    1/8  ✔
            # [8, 128, 30, 40]   1/16 ✔
            # [8, 160, 15, 20]   1/32 ✔
            features = self.rgb_features_extractor(x)

        return features

    def forward(self, x: Tensor, speed_test=False):
        if not speed_test:
            # t0 = time.time()
            # NOTE: 提取RGB的context特征信息
            rgb_features = self.rgb_feature_extractor(x)
            # for i in range(len(rgb_features)):
            #     print("rgb_features[{}].shape:{}".format(i, rgb_features[i].shape))
            # exit()
            # t1 = time.time() - t0

            # NOTE: MobileNetV2-TP_TopFormer_OutputHead_07-30
            #       MobileNetV2-TP_TopFormer_OutputHead_09-14
            # rgb_features = self.rgb_feature_mhsa(rgb_features)
            # y = self.decoder(rgb_features)

            # NOTE: MobileNetV2-TP_BMSAE3_ENV3_EMSAE1_OHV3_07-31
            #       MobileNetV2-TP_BMSAE3_ENV3_EMSAE1_OHV3_kitti_09-06
            # rgb_features = self.rgb_feature_mhsa(rgb_features)  # NOTE: 对RGB特征做MHSA
            # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
            # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
            # edge_features = self.edge_feature_mhsa(edge_features)  # NOTE: 对edge特征做MHSA
            # y = self.decoder(rgb_features, edge_features)  # NOTE: 上采样稠密深度估计

            # NOTE: MobileNetV2-TP_ENV3M1_TMV1_OHV1_08-01
            #       MobileNetV2-TP_ENV3M1_TMV1_OHV1M1_08-02
            #       MobileNetV2-TP_ENV3M1_TMV1_alpha1.0_OHV1_08-02
            #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_08-02
            #       MobileNetV2-TP_ENV3M1_TMV1M1_alpha0.5_OHV1_08-02
            #       MobileNetV2-TP_kitti_ENV3M1_TMV2_alpha0.5_OHV1_08-07
            #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_08-02
            #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_08-30
            #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_09-03
            #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_09-03
            #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_09-05
            #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_09-06
            # # t0 = time.time()
            # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
            # # t2 = time.time()
            # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
            # # t3 = time.time()
            # fused_features = self.transition_module(rgb_features, edge_features)  # NOTE: 融合RGB和edge信息过渡给decoder
            # # t4 = time.time()
            # y = self.decoder(fused_features)
            # # t5 = time.time()

            # print('rgb_feature_extractor time: {:.4f}'.format(t1))
            # print('edge_extractor_torch time: {:.4f}'.format(t2-t0))
            # print('edge_feature_extractor time: {:.4f}'.format(t3-t2))
            # print('transition_module time: {:.4f}'.format(t4-t3))
            # print('decoder time: {:.4f}'.format(t5-t4))

            # rgb = x.detach().cpu()  # [b, 3, h, w]
            # x_edge = x_edge.detach().cpu()  # [b, 3, h, w]
            # rgb = rgb.permute(0, 2, 3, 1)  # [b, h, w, 3]
            # x_edge = x_edge.permute(0, 2, 3, 1)  # [b, h, w, 3]
            # rgb = rgb.numpy()
            # x_edge = x_edge.numpy()

            # rgb = x.detach().cpu().numpy()  # [b, 3, h, w]
            # x_edge = x_edge.detach().cpu().numpy()  # [b, 3, h, w]
            # rgb = rgb.transpose(0, 2, 3, 1)  # [b, h, w, 3]
            # x_edge = x_edge.transpose(0, 2, 3, 1)  # [b, h, w, 3]
            #
            # fig, axes = plt.subplots(1, 2)
            # # 在第一个子图中显示image1
            # axes[0].imshow(rgb[0], cmap='gray')
            # axes[0].set_title('Image 1')
            # # 在第二个子图中显示image2
            # axes[1].imshow(x_edge[0], cmap='gray')
            # axes[1].set_title('Image 2')
            #
            # plt.tight_layout()  # 调整子图之间的间距
            # plt.show()
            # exit()

            # NOTE: MobileNetV2-TP_ENV3M1_EHV1_TMV2_OHV1_08-30
            #       MobileNetV2-TP_ENV3M1_EHV1_TMV2_OHV1_kitti_09-15
            # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
            # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
            # fused_features = self.transition_module(rgb_features, edge_features)  # NOTE: 融合RGB和edge信息过渡给decoder
            # y = self.decoder(fused_features)
            # if self.usage_mode == 'train':
            #     pred_edge = self.edge_head(edge_features)
            #     return y, pred_edge

            # NOTE: MobileNetV2-TP_ENV3M1_TMV3_alpha0.5_OHV4_08-03
            #       MobileNetV2-TP_ENV3M1_TMV3M1_alpha0.5_OHV4_08-04
            #       MobileNetV2-TP_ENV3M1_TMV3M2_alpha0.5_OHV4_08-05
            # rgb_features_ = rgb_features[::-1][:3]  # 倒序取出每层特征 并只取最后三层
            # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
            # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
            # edge_features_ = edge_features[::-1][:3]  # 倒序取出每层特征 并只取最后三层
            # fused_features = self.transition_module(rgb_features_, edge_features_)  # NOTE: 融合RGB和edge信息过渡给decoder
            # fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
            # y = self.decoder(fused_features)

            # NOTE: MobileNetV2-TP_TST_OHV5_08-10
            #       MobileNetV2-TP_TST_OHV5_09-02
            # rgb_features = rgb_features[1:]  # [1/8, 1/16, 1/32]
            # fused_features = self.transition_module(rgb_features)
            # y = self.decoder(fused_features)

            # NOTE: MobileNetV2-TP_SideRT_OHV6_08-11
            # rgb_features_ = rgb_features[::-1]
            # fused_features = self.transition_module(rgb_features_)
            # y = self.decoder(fused_features)

            # NOTE: MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_08-26
            #       MobileNetV2-TP_EPV1_TMV3M1_alpha0.5_OHV4_08-27
            #       MobileNetV2-TP_EPV2_TMV3M1_alpha0.5_OHV4_08-29
            #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_09-03
            #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_kitti_09-04
            #       MobileNetV2-TP_ENV4_TMV3M3_alpha0.5_OHV4_09-04
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_09-04
            #       MobileNetV2-TP_ENV4_TMV3M5_alpha0.5_OHV4_09-04
            #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_09-04
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti_09-05
            #       MobileNetV2-TP_ENV4_TMV3M6_LearnableAlpha_OHV4_09-06
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_09-26
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_09-27
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti_09-27
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_09-28
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-09
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-10
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-13
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-bts_10-16
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-16
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-17
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-bts_10-17
            #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-18
            rgb_features_ = rgb_features[::-1][:3]  # 倒序取出每层特征 并只取最后三层
            x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
            edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
            fused_features = self.transition_module(rgb_features_, edge_features)  # NOTE: 融合RGB和edge信息过渡给decoder
            # fused_features, attn_score_list, x_list = self.transition_module(rgb_features_, edge_features)
            fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
            y = self.decoder(fused_features)

            # NOTE: MobileNetV2-TP_ENV4M1_TMV3M1_OHV7_09-03
            # rgb_features_ = rgb_features[::-1][:3]  # 倒序取出每层特征 并只取最后三层
            # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
            # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
            # edge_features_ = edge_features[:3]  # NOTE: 只取前三层1/8 1/16 1/32分辨率
            # fused_features = self.transition_module(rgb_features_, edge_features_)  # NOTE: 融合RGB和edge信息过渡给decoder
            # fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
            # y = self.decoder(fused_features, edge_features)

            # NOTE: MobileNetV2-TP_SGN_TMV3M4_LearnableAlpha_OHV4_09-26
            #       MobileNetV2-TP_SGN_TMV3M4_ConstAlpha_OHV4_09-27
            #       MobileNetV2-TP_SGN_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_09-28
            #       MobileNetV2-TP_SGN_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-09
            # guidance_features = self.guidance_feature_extractor(x)
            # rgb_features_ = rgb_features[::-1][:3]  # 倒序取出每层特征 并只取最后三层
            # fused_features = self.transition_module(rgb_features_, guidance_features)
            # fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
            # y = self.decoder(fused_features)





            # # NOTE: 对RGB特征做MHSA
            # rgb_features = self.rgb_feature_mhsa(rgb_features)

            # # NOTE: 提取RGB的edge
            # # x_edge = edge_extractor(x, 'sobel')
            # x_edge = self.edge_extractor_torch(x, device=self.device)
            #
            # # NOTE: 提取edge的特征信息
            # # [B, 16, 240, 320]  !
            # # [B, 24, 120, 160]  !
            # # [B, 32, 60, 80]    !
            # # [B, 64, 30, 40]    !
            # # [B, 96, 30, 40]    !
            # edge_features = self.edge_feature_extractor(x_edge)
            # # for i in range(len(edge_features)):
            # #     print("edge_features[{}].shape:{}".format(i, edge_features[i].shape))
            #
            # # NOTE: 对edge特征做MHSA
            # edge_features = self.edge_feature_mhsa(edge_features)
            # # exit()
            #
            # # NOTE: 融合RGB和edge信息过渡给decoder
            # # rgb_features = self.transition_module(rgb_features, edge_features)
            # # for i in range(len(rgb_features)):
            # #     print("rgb_features[{}].shape:{}".format(i, rgb_features[i].shape))
            # # exit()
            #
            # # NOTE: 上采样稠密深度估计
            # # y = self.decoder(rgb_features, edge_features)
            # # y = self.decoder(rgb_features[0], rgb_features[1], rgb_features[2], rgb_features[3])
            # # y = self.decoder(rgb_features)
            # y = self.decoder(rgb_features, edge_features)

            return y  # attn_score_list, x_list  # y rgb_features

        else:
            # times = []
            times = {}
            t0 = time.time()
            # NOTE: 提取RGB的context特征信息
            rgb_features = self.rgb_feature_extractor(x)
            # print('rgb_feature_extractor time: {:.6f}s'.format(times[-1]))

            # NOTE: MobileNetV2-TP_BMSAE3_ENV3_EMSAE1_OHV3_07-31
            # rgb_features = self.rgb_feature_mhsa(rgb_features)  # NOTE: 对RGB特征做MHSA
            # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
            # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
            # edge_features = self.edge_feature_mhsa(edge_features)  # NOTE: 对edge特征做MHSA
            # y = self.decoder(rgb_features, edge_features)  # NOTE: 上采样稠密深度估计

            # NOTE: MobileNetV2-TP_ENV3M1_TMV1_OHV1_08-01
            #       MobileNetV2-TP_ENV3M1_TMV1_OHV1M1_08-02
            #       MobileNetV2-TP_ENV3M1_TMV1_alpha1.0_OHV1_08-02
            #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_08-02
            #       MobileNetV2-TP_ENV3M1_TMV1M1_alpha0.5_OHV1_08-02
            #       MobileNetV2-TP_kitti_ENV3M1_TMV2_alpha0.5_OHV1_08-07
            t1 = time.time()
            x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
            t2 = time.time()
            edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
            t3 = time.time()
            fused_features = self.transition_module(rgb_features, edge_features)  # NOTE: 融合RGB和edge信息过渡给decoder
            t4 = time.time()
            y = self.decoder(fused_features)
            t5 = time.time()

            # times.append(t1 - t0)
            # times.append(t2 - t1)
            # times.append(t3 - t2)
            # times.append(t4 - t3)
            # times.append(t5 - t4)

            times['rgb_feature_extractor'] = t1 - t0
            times['edge_extractor_torch'] = t2 - t1
            times['edge_feature_extractor'] = t3 - t2
            times['transition_module'] = t4 - t3
            times['decoder'] = t5 - t4

            # NOTE: MobileNetV2-TP_ENV3M1_TMV3_alpha0.5_OHV4_08-03
            #       MobileNetV2-TP_ENV3M1_TMV3M1_alpha0.5_OHV4_08-04
            #       MobileNetV2-TP_ENV3M1_TMV3M2_alpha0.5_OHV4_08-05
            # rgb_features_ = rgb_features[::-1][:3]  # 倒序取出每层特征 并只取最后三层
            # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
            # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
            # edge_features_ = edge_features[::-1][:3]  # 倒序取出每层特征 并只取最后三层
            # fused_features = self.transition_module(rgb_features_, edge_features_)  # NOTE: 融合RGB和edge信息过渡给decoder
            # fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
            # y = self.decoder(fused_features)

            # NOTE: MobileNetV2-TP_TST_OHV5_08-10
            # rgb_features = rgb_features[1:]  # [1/8, 1/16, 1/32]
            # fused_features = self.transition_module(rgb_features)
            # y = self.decoder(fused_features)

            # NOTE: MobileNetV2-TP_SideRT_OHV6_08-11
            # rgb_features_ = rgb_features[::-1]
            # t0 = time.time()
            # fused_features = self.transition_module(rgb_features_)
            # times.append(time.time() - t0)
            # print('transition_module time: {:.6f}s'.format(times[-1]))
            # t0 = time.time()
            # y = self.decoder(fused_features)
            # times.append(time.time() - t0)
            # print('decoder time: {:.6f}s'.format(times[-1]))

            return y, times

    # def edge_extractor_torch(self, x: Tensor, device):
    #     conv_rgb_core_sobel_horizontal = [
    #         [[1, 2, 1], [0, 0, 0], [-1, -2, -1],
    #          [0, 0, 0], [0, 0, 0], [0, 0, 0],
    #          [0, 0, 0], [0, 0, 0], [0, 0, 0]
    #          ],
    #         [[0, 0, 0], [0, 0, 0], [0, 0, 0],
    #          [1, 2, 1], [0, 0, 0], [-1, -2, -1],
    #          [0, 0, 0], [0, 0, 0], [0, 0, 0]
    #          ],
    #         [[0, 0, 0], [0, 0, 0], [0, 0, 0],
    #          [0, 0, 0], [0, 0, 0], [0, 0, 0],
    #          [1, 2, 1], [0, 0, 0], [-1, -2, -1],
    #          ]]
    #     conv_rgb_core_sobel_vertical = [
    #         [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
    #          [0, 0, 0], [0, 0, 0], [0, 0, 0],
    #          [0, 0, 0], [0, 0, 0], [0, 0, 0]
    #          ],
    #         [[0, 0, 0], [0, 0, 0], [0, 0, 0],
    #          [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
    #          [0, 0, 0], [0, 0, 0], [0, 0, 0]
    #          ],
    #         [[0, 0, 0], [0, 0, 0], [0, 0, 0],
    #          [0, 0, 0], [0, 0, 0], [0, 0, 0],
    #          [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
    #          ]]
    #
    #     conv_op_horizontal = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
    #     conv_op_vertical = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
    #
    #     sobel_kernel_horizontal = np.array(conv_rgb_core_sobel_horizontal, dtype='float32')  # float32
    #     sobel_kernel_horizontal = sobel_kernel_horizontal.reshape((3, 3, 3, 3))
    #     conv_op_horizontal.weight.data = torch.from_numpy(sobel_kernel_horizontal).to(device)
    #
    #     sobel_kernel_vertical = np.array(conv_rgb_core_sobel_vertical, dtype='float32')  # float32
    #     sobel_kernel_vertical = sobel_kernel_vertical.reshape((3, 3, 3, 3))
    #     conv_op_vertical.weight.data = torch.from_numpy(sobel_kernel_vertical).to(device)
    #
    #     sobel_x = conv_op_horizontal(x)
    #     sobel_y = conv_op_vertical(x)
    #     sobel_xy = torch.sqrt(torch.square(sobel_x) + torch.square(sobel_y))
    #
    #     # sobel_x = normalize2img_tensor(sobel_x)
    #     # sobel_y = normalize2img_tensor(sobel_y)
    #     sobel_xy = normalize2img_tensor(sobel_xy)  # NOTE: 原来版本  domain: [0, 255]
    #     # sobel_xy = self.normalize_to_01(sobel_xy)  # NOTE: 当前测试版  严重影响推理速度 因为有for操作
    #
    #     return sobel_xy  # sobel_x, sobel_y, sobel_xy

    def edge_extractor_torch(self, x: Tensor, device):
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

        # sobel_x = self.conv_op_horizontal(x)
        # sobel_y = self.conv_op_vertical(x)

        sobel_xy = torch.sqrt(torch.square(sobel_x) + torch.square(sobel_y))

        # NOTE: 直接不做归一化处理 因为EdgeNet中自带BatchNorm层会归一化数值
        # sobel_x = normalize2img_tensor(sobel_x)
        # sobel_y = normalize2img_tensor(sobel_y)
        # NOTE: 原来版本  domain: [0, 255]
        #       MobileNetV2-TP_BMSAE3_ENV3_EMSAE1_OHV3_07-31
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_08-02
        #       MobileNetV2-TP_ENV3M1_TMV3M1_alpha0.5_OHV4_08-04
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_09-05
        #       MobileNetV2-TP_BMSAE3_ENV3_EMSAE1_OHV3_kitti_09-06
        # sobel_xy = normalize2img_tensor(sobel_xy)

        # NOTE: 当前测试版  严重影响推理速度 因为有for操作
        #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_08-26
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_08-30
        #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_09-03
        #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_kitti_09-04
        #       MobileNetV2-TP_ENV4_TMV3M3_alpha0.5_OHV4_09-04
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_09-04
        #       MobileNetV2-TP_ENV4_TMV3M5_alpha0.5_OHV4_09-04
        #       MobileNetV2-TP_ENV4_TMV3M1_alpha0.5_OHV4_09-04
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti_09-05
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_kitti_09-06
        #       MobileNetV2-TP_ENV4_TMV3M6_LearnableAlpha_OHV4_09-06
        #       MobileNetV2-TP_ENV3M1_EHV1_TMV2_OHV1_08-30
        #       MobileNetV2-TP_ENV3M1_EHV1_TMV2_OHV1_kitti_09-15
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_09-26
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_09-27
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti_09-27
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_09-28
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-09
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-10
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-13
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-bts_10-16
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-16
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-17
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_kitti-bts_10-17
        #       MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_cosineLR_10-18
        sobel_xy = self.normalize_to_01(sobel_xy)

        # NOTE:
        #       MobileNetV2-TP_ENV4M1_TMV3M1_OHV7_09-03
        #       MobileNetV2-TP_ENV3M1_TMV2_alpha0.5_OHV1_09-03
        # sobel_xy = self.normalize_to_01_torch(sobel_xy)

        return sobel_xy  # sobel_x, sobel_y, sobel_xy

    def normalize_to_01_torch(self, x):
        b, c, h, w = x.shape
        x_ = x.reshape(b, c, h * w)
        # 将hw展平 在每个batch的三通道边缘图的每个通道上找最值 对每个通道做归一化
        vmax = torch.max(x_, dim=-1, keepdim=True)
        vmin = torch.min(x_, dim=-1, keepdim=True)
        vmax = vmax.values  # [b, c, 1]
        vmin = vmin.values
        vmax = vmax.repeat(1, 1, h * w).reshape(b, c, h, w)
        vmin = vmin.repeat(1, 1, h * w).reshape(b, c, h, w)
        x = (x - vmin) / (vmax - vmin)
        # print('temp1 max:{} min:{}'.format(torch.max(x[0, 0:1, :, :]), torch.min(x[0, 0:1, :, :])))
        # print('temp2 max:{} min:{}'.format(torch.max(x[0, 1:2, :, :]), torch.min(x[0, 1:2, :, :])))
        # print('temp3 max:{} min:{}'.format(torch.max(x[0, 2:, :, :]), torch.min(x[0, 2:, :, :])))
        return x

    def normalize2img_tensor(self, x: Tensor):  # NOTE: 原来版本
        min_val = x.min()
        max_val = x.max()
        res = (x - min_val) / (max_val - min_val)
        res = res * 255.
        return res

    def normalize_to_01(self, x):
        # print('x.size()[1]:{}'.format(x.size()[1]))
        channel_list = []
        for i in range(x.size()[1]):
            temp = x[:, i:i+1, :, :]
            vmax = torch.max(temp).item()
            vmin = torch.min(temp).item()
            temp = (temp - vmin) / (vmax - vmin)
            # print('temp.shape:{}'.format(temp.shape))
            channel_list.append(temp)
        # print('len(channel_list):{}'.format(len(channel_list)))
        res = torch.cat(channel_list, dim=1)
        # print('res.shape:{} max:{} min:{}'.format(res.shape, torch.max(res), torch.min(res)))
        # print('res[0]:\n{}'.format(res[0]))
        return res

    def _make_extension_module(
        self,
        opts,
        input_channel: int,  # paper中为96
    ):
        extension_module = nn.Sequential()
        expansion_factor = 4
        stride = 1
        output_channel = 128
        # out_channels = 128
        num_blocks = 6
        width_mult = 1.0
        dilation_rates = [1, 2, 3, 1, 2, 3]

        # output_channel = make_divisible(out_channels * width_mult, self.round_nearest)

        for i in range(num_blocks):
            block_name = "extension_module_IRB_{}".format(i)
            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channel,
                stride=stride,
                expand_ratio=expansion_factor,
                dilation=dilation_rates[i],
            )
            extension_module.add_module(name=block_name, module=layer)
            input_channel = output_channel

        return extension_module

    def _make_layer(
            self,
            opts,
            mv2_config: Dict or List,
            width_mult: float,
            input_channel: int,
            dilate: Optional[bool] = False,
            *args,
            **kwargs
    ) -> Tuple[nn.Module, int]:
        prev_dilation = self.dilation
        mv2_block = nn.Sequential()
        count = 0

        if isinstance(mv2_config, Dict):
            mv2_config = [mv2_config]

        for cfg in mv2_config:
            t = cfg.get("expansion_ratio")
            c = cfg.get("out_channels")
            n = cfg.get("num_blocks")
            s = cfg.get("stride")

            output_channel = make_divisible(c * width_mult, self.round_nearest)

            for block_idx in range(n):
                stride = s if block_idx == 0 else 1
                block_name = "mv2_block_{}".format(count)
                if dilate and count == 0:
                    self.dilation *= stride
                    stride = 1

                layer = InvertedResidual(
                    opts=opts,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=stride,
                    expand_ratio=t,
                    dilation=prev_dilation if count == 0 else self.dilation,
                )
                mv2_block.add_module(name=block_name, module=layer)
                count += 1
                input_channel = output_channel
        return mv2_block, input_channel


