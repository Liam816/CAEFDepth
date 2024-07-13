import torch
import torchvision
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os
import numpy as np
import torch.nn.functional as F

from utils import logger
from ..modules import PConvGuidedUpsampleBlock
from ..modules.se import SELayer
from ..modules.cbam import CBAM
from ..modules.eca_module import eca_layer
from ..modules.srm_module import SRMLayer
from ..modules.gct_module import GCTLayer
# from ..layers.qna import QnAStage
# from ..modules.tf_block_raw import TFBlock
from ..modules.tf_block_raw import TFPatchEmbed, Attention, TFDropPath, PatchEmbedUnfold, MyAttention
from ..modules.tf_block_topformer import TFBlock, BasicLayer, SemanticInjectionModule, LTG, GTG, RRDecoderV1
from .MobileNetV2_LiamEdge import EdgeNetV4 as ENV4
from .MobileNetV2_LiamEdge import TransitionModuleV3M4, OutputHeadV4
import time


def normalize2img_tensor(x: Tensor):
    min_val = x.min()
    max_val = x.max()
    res = (x - min_val) / (max_val - min_val)
    res = res * 255.
    return res


class EdgeNetV1(nn.Module):
    def __init__(self):
        super(EdgeNetV1, self).__init__()

        # NOTE: Params(0.25M)
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


class EdgeNetV1_4L(nn.Module):
    def __init__(self):
        super(EdgeNetV1_4L, self).__init__()

        # NOTE: Params(0.25M)
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # kernel: [8, 3, 3, 3]
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # kernel: [16, 8, 3, 3]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/4, w/4]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/8, w/8]

        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )  # [b, 256, h/16, w/16]

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 32, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.edge_encoder_3(x)  # [B, 128, 60, 80]
        features.append(x)
        x = self.edge_encoder_4(x)  # [B, 256, 30, 40]
        features.append(x)
        return features


class EdgeNetV1_4L_N(nn.Module):
    def __init__(self):
        super(EdgeNetV1_4L_N, self).__init__()

        # NOTE: Params(0.25M)
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # kernel: [8, 3, 3, 3]
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # kernel: [16, 8, 3, 3]
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 32, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.edge_encoder_3(x)  # [B, 128, 60, 80]
        features.append(x)
        x = self.edge_encoder_4(x)  # [B, 256, 30, 40]
        features.append(x)
        return features


class EdgeNetV2(nn.Module):
    def __init__(self):
        super(EdgeNetV2, self).__init__()

        # NOTE: Params(0.25M)
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/4, w/4]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/16, w/16]

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.edge_encoder_3(x)  # [B, 64, 60, 80]
        features.append(x)
        return features


class EdgeNetV2N(nn.Module):
    def __init__(self):
        super(EdgeNetV2N, self).__init__()

        # NOTE: Params(0.25M)
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )  # [b, 16, h/4, w/4]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/8, w/8]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/16, w/16]

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)
        features.append(x)
        x = self.edge_encoder_2(x)
        features.append(x)
        x = self.edge_encoder_3(x)
        features.append(x)
        return features


class EdgeNetV2Attn(nn.Module):
    def __init__(self, attn_type=None):
        super(EdgeNetV2Attn, self).__init__()
        self.attn_type = attn_type
        # NOTE: Params(0.25M)
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/4, w/4]
        if attn_type == 'cbam':
            self.attn_1 = CBAM(32)
        elif attn_type == 'cbams':
            self.attn_1 = CBAM(32, no_channel=True)
        elif attn_type == 'cbamc':
            self.attn_1 = CBAM(32, no_spatial=True)

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]
        if attn_type == 'cbam':
            self.attn_2 = CBAM(64)
        elif attn_type == 'cbams':
            self.attn_2 = CBAM(64, no_channel=True)
        elif attn_type == 'cbamc':
            self.attn_2 = CBAM(64, no_spatial=True)

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/16, w/16]
        if attn_type == 'cbam':
            self.attn_3 = CBAM(128)
        elif attn_type == 'cbams':
            self.attn_3 = CBAM(128, no_channel=True)
        elif attn_type == 'cbamc':
            self.attn_3 = CBAM(128, no_spatial=True)

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        if self.attn_type is not None:
            x = self.attn_1(x)
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
        if self.attn_type is not None:
            x = self.attn_2(x)
        features.append(x)
        x = self.edge_encoder_3(x)  # [B, 64, 60, 80]
        if self.attn_type is not None:
            x = self.attn_3(x)
        features.append(x)
        return features


# class BasicLayer(nn.Module):
#     """
#     Scale-Aware Semantics Extractor
#     """
#     def __init__(self, block_num, embedding_dim, key_dim, num_heads,
#                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
#                 norm_cfg=dict(type='BN2d', requires_grad=True),
#                 act_layer=None):
#         super().__init__()
#         self.block_num = block_num  # 默认为4
#
#         self.transformer_blocks = nn.ModuleList()
#         for i in range(self.block_num):
#             self.transformer_blocks.append(TFBlock(
#                 embedding_dim, key_dim=key_dim, num_heads=num_heads,
#                 mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
#                 drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_cfg=norm_cfg,
#                 act_layer=act_layer))
#
#     def forward(self, x):
#         # token * N
#         for i in range(self.block_num):
#             x = self.transformer_blocks[i](x)
#         return x


# class EdgeNetV2TF(nn.Module):
#     def __init__(self, shape=None):
#         super(EdgeNetV2TF, self).__init__()
#
#         # h, w = shape[0], shape[1]
#
#         depths = 4
#         drop_path_rate = 0.1
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
#         # norm_cfg = dict(type='SyncBN', requires_grad=True)
#         norm_cfg = dict(type='BN', requires_grad=True)
#         act_layer = nn.ReLU6
#
#         # NOTE: Params(0.25M)
#         self.edge_encoder_1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )  # [b, 32, h/4, w/4]
#
#         # self.tf_block = TFBlock(32, key_dim=16, num_heads=8,
#         #                         drop=0, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#         #                         norm_cfg=norm_cfg, act_layer=act_layer)
#         self.tf_layer1 = BasicLayer(
#             block_num=depths,
#             embedding_dim=32,  # 承接其上的卷积层输出的通道数
#             key_dim=16,
#             num_heads=8,
#             mlp_ratio=2,
#             attn_ratio=2,
#             drop=0, attn_drop=0,
#             drop_path=dpr,
#             norm_cfg=norm_cfg,
#             act_layer=act_layer
#         )
#
#         self.edge_encoder_2 = nn.Sequential(
#             nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
#             nn.BatchNorm2d(48),
#             nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )  # [b, 64, h/8, w/8]
#
#         self.tf_layer2 = BasicLayer(
#             block_num=depths,
#             embedding_dim=64,  # 承接其上的卷积层输出的通道数
#             key_dim=16,
#             num_heads=8,
#             mlp_ratio=2,
#             attn_ratio=2,
#             drop=0, attn_drop=0,
#             drop_path=dpr,
#             norm_cfg=norm_cfg,
#             act_layer=act_layer
#         )
#
#         self.edge_encoder_3 = nn.Sequential(
#             nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
#             nn.BatchNorm2d(96),
#             nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )  # [b, 128, h/16, w/16]
#
#         self.tf_layer3 = BasicLayer(
#             block_num=depths,
#             embedding_dim=128,  # 承接其上的卷积层输出的通道数
#             key_dim=16,
#             num_heads=8,
#             mlp_ratio=2,
#             attn_ratio=2,
#             drop=0, attn_drop=0,
#             drop_path=dpr,
#             norm_cfg=norm_cfg,
#             act_layer=act_layer
#         )
#
#     def forward(self, x):
#         features = []
#         x = self.edge_encoder_1(x)  # [B, 16, 240, 320]
#         print('x.shape:{}'.format(x.shape))
#         x = self.tf_layer1(x)
#         print('x.shape:{}'.format(x.shape))
#         features.append(x)
#
#         x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
#         print('x.shape:{}'.format(x.shape))
#         x = self.tf_layer2(x)
#         print('x.shape:{}'.format(x.shape))
#         features.append(x)
#
#         x = self.edge_encoder_3(x)  # [B, 64, 60, 80]
#         print('x.shape:{}'.format(x.shape))
#         x = self.tf_layer3(x)
#         print('x.shape:{}'.format(x.shape))
#         features.append(x)
#
#         return features


class EdgeNetV2TF(nn.Module):
    def __init__(self, img_shape=(480, 640)):
        super(EdgeNetV2TF, self).__init__()

        h, w = img_shape[0], img_shape[1]
        # print('h:{}, w:{}'.format(h, w))

        # NOTE: Params(0.25M)
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/4, w/4]

        self.drop_path_1 = TFDropPath(0.0333)
        ps = 4
        conv_output_channels = 32
        self.embed_1 = TFPatchEmbed(img_size=(h//4, w//4), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_1 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_1 = MyAttention(dim=ps*ps*conv_output_channels, visualization=False)  # Attention MyAttention
        self.unfold_1 = PatchEmbedUnfold(feature_map_size=(h//4, w//4), patch_size=ps)

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]

        self.drop_path_2 = TFDropPath(0.0666)
        ps = 2
        conv_output_channels = 64
        self.embed_2 = TFPatchEmbed(img_size=(h//8, w//8), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_2 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_2 = MyAttention(dim=ps*ps*conv_output_channels, visualization=False)  # Attention MyAttention
        self.unfold_2 = PatchEmbedUnfold(feature_map_size=(h//8, w//8), patch_size=ps)

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/16, w/16]

        self.drop_path_3 = TFDropPath(0.1000)
        ps = 1
        conv_output_channels = 128
        self.embed_3 = TFPatchEmbed(img_size=(h//16, w//16), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_3 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_3 = MyAttention(dim=ps*ps*conv_output_channels, visualization=False)  # Attention MyAttention
        self.unfold_3 = PatchEmbedUnfold(feature_map_size=(h//16, w//16), patch_size=ps)

    def forward(self, x):
        features = []
        x1 = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        x2 = self.edge_encoder_2(x1)  # [B, 64, 120, 160]
        x3 = self.edge_encoder_3(x2)  # [B, 64, 60, 80]

        t0 = time.time()
        x1 = self.embed_1(x1)
        x1_ = x1 + self.drop_path_1(self.mhsa_1(x1))
        x2 = self.embed_2(x2)
        x2_ = x2 + self.drop_path_2(self.mhsa_2(x2))
        x3 = self.embed_3(x3)
        x3_ = x3 + self.drop_path_3(self.mhsa_3(x3))
        # print('attn time:{:.6f}'.format(time.time() - t0))

        # print('x1_.shape:', x1_.shape)
        # print('x2_.shape:', x2_.shape)
        # print('x3_.shape:', x3_.shape)

        t0 = time.time()
        x1_ = self.unfold_1(x1_)
        x2_ = self.unfold_2(x2_)
        x3_ = self.unfold_3(x3_)
        # print('unfold time:{:.6f}'.format(time.time() - t0))

        # print('x1_.shape:', x1_.shape)
        # print('x2_.shape:', x2_.shape)
        # print('x3_.shape:', x3_.shape)

        # exit()

        x1_ = F.interpolate(x1_, scale_factor=4.0, mode='bilinear')
        x2_ = F.interpolate(x2_, scale_factor=2.0, mode='bilinear')

        # print('x1_.shape:', x1_.shape)
        # print('x2_.shape:', x2_.shape)
        # print('x3_.shape:', x3_.shape)
        #
        # exit()

        features.append(x1_)
        features.append(x2_)
        features.append(x3_)

        return features

    # def forward(self, x):
    #     features = []
    #     x1 = self.edge_encoder_1(x)  # [B, 16, 240, 320]
    #     x2 = self.edge_encoder_2(x1)  # [B, 64, 120, 160]
    #     x3 = self.edge_encoder_3(x2)  # [B, 64, 60, 80]
    #
    #     # temp1 = []
    #     # temp1.append(x1)
    #     # temp1.append(x2)
    #     # temp1.append(x3)
    #
    #     t0 = time.time()
    #     x1 = self.embed_1(x1)
    #     x1, attn_sc_1 = self.mhsa_1(x1)
    #     x1_ = x1 + self.drop_path_1(x1)
    #
    #     x2 = self.embed_2(x2)
    #     x2, attn_sc_2 = self.mhsa_2(x2)
    #     x2_ = x2 + self.drop_path_2(x2)
    #
    #     x3 = self.embed_3(x3)
    #     x3, attn_sc_3 = self.mhsa_3(x3)
    #     x3_ = x3 + self.drop_path_3(x3)
    #     # print('attn time:{:.6f}'.format(time.time() - t0))
    #
    #     # print('attn_sc_1.shape:{}'.format(attn_sc_1.shape))
    #     # print('attn_sc_2.shape:{}'.format(attn_sc_2.shape))
    #     # print('attn_sc_3.shape:{}'.format(attn_sc_3.shape))
    #     # exit()
    #
    #     t0 = time.time()
    #     x1_ = self.unfold_1(x1_)
    #     x2_ = self.unfold_2(x2_)
    #     x3_ = self.unfold_3(x3_)
    #     # print('unfold time:{:.6f}'.format(time.time() - t0))
    #
    #     x1_ = F.interpolate(x1_, scale_factor=4.0, mode='bilinear')
    #     x2_ = F.interpolate(x2_, scale_factor=2.0, mode='bilinear')
    #
    #     # temp2 = []
    #     # temp2.append(x1_)
    #     # temp2.append(x2_)
    #     # temp2.append(x3_)
    #
    #     features.append(x1_)
    #     features.append(x2_)
    #     features.append(x3_)
    #
    #     return features  # temp1, temp2


class EdgeNetV3(nn.Module):
    def __init__(self):
        super(EdgeNetV3, self).__init__()

        # NOTE: Params(M)
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )  # [b, 16, h/2, w/2]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/4, w/4]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]

        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/16, w/16]

        # self.edge_encoder_1 = nn.Sequential(
        #     nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True),
        # )  # [b, 8, h/2, w/2]
        #
        # self.edge_encoder_2 = nn.Sequential(
        #     nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        # )  # [b, 16, h/4, w/4]
        #
        # self.edge_encoder_3 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        # )  # [b, 32, h/8, w/8]
        #
        # self.edge_encoder_4 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )  # [b, 64, h/16, w/16]

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


class EdgeNetV4(nn.Module):
    def __init__(self):
        super(EdgeNetV4, self).__init__()

        # NOTE: Params(M)
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )  # [b, 8, h/2, w/2]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )  # [b, 16, h/4, w/4]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/8, w/8]

        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/16, w/16]

        self.edge_encoder_5 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
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
        x = self.edge_encoder_5(x)
        features.append(x)
        return features


class EdgeNetV5(nn.Module):
    def __init__(self, resolution):
        super(EdgeNetV5, self).__init__()
        h, w = resolution[0], resolution[1]

        # NOTE: Params(0.25M)
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [b, 32, h, w]

        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # [b, 64, h, w]

        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # [b, 128, h, w]

        self.oct_res_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.quarter_res_conv = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)

        # self.down_sample_quater = torchvision.transforms.Resize((h/4, w/4))
        # self.down_sample_oct = torchvision.transforms.Resize((h/8, w/8))

    def forward(self, x):
        # x_quater = torchvision.transforms.Resize(x.shape[-2:] / 4)
        # x_oct = torchvision.transforms.Resize(x.shape[-2:] / 8)
        #
        # features = []
        #
        # x = self.edge_encoder_1(x)
        # x = self.edge_encoder_2(x)
        # x = self.edge_encoder_3(x)
        # x = self.full_res_conv(x)
        # features.append(x)  # [b, 32, h, w]
        #
        # x_quater = self.edge_encoder_1(x_quater)
        # x_quater = self.edge_encoder_2(x_quater)
        # x_quater = self.edge_encoder_3(x_quater)
        # x_quater = self.quarter_res_conv(x_quater)
        # features.append(x_quater)  # [b, 64, h/4, w/4]
        #
        # x_oct = self.edge_encoder_1(x_oct)
        # x_oct = self.edge_encoder_2(x_oct)
        # x_oct = self.edge_encoder_3(x_oct)
        # features.append(x_oct)  # [b, 128, h/8, w/8]

        # # 四分之一尺寸
        # if x.shape[-1] == (self.resolution[1] / 4) and x.shape[-2] == (self.resolution[0] / 4):
        #     x = self.quarter_res_conv(x)
        # # 完全尺寸
        # elif x.shape[-1] == self.resolution[1] and x.shape[-2] == self.resolution[0]:
        #     x = self.full_res_conv(x)

        # h, w = x.size()[-2], x.size()[-1]
        # raw_shape = (h, w)
        # print('raw_shape:{}'.format(raw_shape))

        features = []
        x = self.edge_encoder_1(x)
        x = self.edge_encoder_2(x)
        x = self.edge_encoder_3(x)

        x_quarter = x
        x_oct = x
        x_sixteenth = x

        x_quarter = self.quarter_res_conv(x_quarter)
        x_quarter = F.interpolate(x_quarter, scale_factor=0.25, mode='bilinear')
        features.append(x_quarter)  # [b, 32, h/4, w/4]

        x_oct = self.oct_res_conv(x_oct)
        x_oct = F.interpolate(x_oct, scale_factor=0.125, mode='bilinear')
        features.append(x_oct)  # [b, 64, h/8, w/8]

        x_sixteenth = F.interpolate(x_sixteenth, scale_factor=0.0625, mode='bilinear')  # [b, 64, h/4, w/4]
        features.append(x_sixteenth)  # [b, 128, h/16, w/16]

        # print('x.shape:{}'.format(x.shape))
        # print('x_quater.shape:{}'.format(x_quater.shape))
        # print('x_oct.shape:{}'.format(x_oct.shape))
        # exit()

        return features


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
        if self.block_sn >= 3:
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


class DecoderV6(nn.Module):
    def __init__(self):
        super(DecoderV6, self).__init__()

        # NOTE: for EdgeNetV1, EdgeNetV3
        self.decoder_block_1 = DecoderBlockV6(1, 256, 64, output_feats=32)
        self.decoder_block_2 = DecoderBlockV6(2, 256, 64, output_feats=32, target_feats=32)
        self.decoder_block_3 = DecoderBlockV6(3, 128, 16, output_feats=1, target_feats=32, is_lastblock=True)

        # NOTE: for EdgeNetV2
        # self.decoder_block_1 = DecoderBlockV6(1, 64, 72, output_feats=32)
        # self.decoder_block_2 = DecoderBlockV6(2, 64, 72, output_feats=32, target_feats=32)
        # self.decoder_block_3 = DecoderBlockV6(3, 32, 18, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)
        # y, temp1, temp2 = self.decoder_block_3(features[0], edges[-3], target=y)

        return y  # , temp1, temp2


class DecoderV6_4L(nn.Module):
    def __init__(self):
        super(DecoderV6_4L, self).__init__()

        # NOTE: for EdgeNetV1
        self.decoder_block_1 = DecoderBlockV6(1, 512, 256, output_feats=256)
        self.decoder_block_2 = DecoderBlockV6(2, 256, 128, output_feats=128, target_feats=256)
        self.decoder_block_3 = DecoderBlockV6(3, 256, 64, output_feats=32, target_feats=128)
        self.decoder_block_4 = DecoderBlockV6(4, 128, 32, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-4], edges[-3], target=y)
        y = self.decoder_block_4(features[-5], edges[-4], target=y)

        return y


class DecoderV6_4L_N(nn.Module):
    def __init__(self):
        super(DecoderV6_4L_N, self).__init__()

        # NOTE: for EdgeNetV1
        self.decoder_block_1 = DecoderBlockV6(1, 512, 128, output_feats=128)
        self.decoder_block_2 = DecoderBlockV6(2, 256, 64, output_feats=64, target_feats=128)
        self.decoder_block_3 = DecoderBlockV6(3, 256, 32, output_feats=32, target_feats=64)
        self.decoder_block_4 = DecoderBlockV6(4, 128, 16, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-4], edges[-3], target=y)
        y = self.decoder_block_4(features[-5], edges[-4], target=y)

        return y


class DecoderBlockV6_woRGBFeats(nn.Module):
    def __init__(self,
                 block_sn,
                 target_feats,
                 edge_feats,
                 output_feats,
                 use_cbam=True,
                 is_lastblock=False):
        super(DecoderBlockV6_woRGBFeats, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam
        self.is_lastblock = is_lastblock

        n_feats = target_feats + edge_feats

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
                nn.Conv2d(edge_feats//2, output_feats, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, target, edge):
        x = torch.cat((target, edge), dim=1)
        x = self.conv1(x)
        if self.use_cbam is True:
            x = self.cbam(x)
        x = x + edge
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        x = self.conv2(x)

        return x


class DecoderV6_4L_woRGBFeats(nn.Module):
    def __init__(self):
        super(DecoderV6_4L_woRGBFeats, self).__init__()

        # NOTE: for EdgeNetV1
        self.decoder_block_1 = DecoderBlockV6_woRGBFeats(1, 512, 256, output_feats=128)
        self.decoder_block_2 = DecoderBlockV6_woRGBFeats(2, 128, 128, output_feats=64)
        self.decoder_block_3 = DecoderBlockV6_woRGBFeats(3, 64, 64, output_feats=32)
        self.decoder_block_4 = DecoderBlockV6_woRGBFeats(4, 32, 32, output_feats=1, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(y, edges[-2])
        y = self.decoder_block_3(y, edges[-3])
        y = self.decoder_block_4(y, edges[-4])

        return y


class DecoderV6_4L_N_woRGBFeats(nn.Module):
    def __init__(self):
        super(DecoderV6_4L_N_woRGBFeats, self).__init__()

        # NOTE: for EdgeNetV1
        self.decoder_block_1 = DecoderBlockV6_woRGBFeats(1, 512, 128, output_feats=128)
        self.decoder_block_2 = DecoderBlockV6_woRGBFeats(2, 128, 64, output_feats=64)
        self.decoder_block_3 = DecoderBlockV6_woRGBFeats(3, 64, 32, output_feats=32)
        self.decoder_block_4 = DecoderBlockV6_woRGBFeats(4, 32, 16, output_feats=1, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(y, edges[-2])
        y = self.decoder_block_3(y, edges[-3])
        y = self.decoder_block_4(y, edges[-4])

        return y


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
        rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.use_cbam is True:
            x = self.cbam(x)

        x = x + edge

        if self.block_sn == 3:
            x = F.interpolate(x, scale_factor=4.0, mode='bilinear')
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')

        x = self.conv2(x)

        return x


class DecoderV7(nn.Module):
    def __init__(self):
        super(DecoderV7, self).__init__()

        self.decoder_block_1 = DecoderBlockV7(1, 256, 128, output_feats=64)
        self.decoder_block_2 = DecoderBlockV7(2, 128, 64, output_feats=32, target_feats=64)
        self.decoder_block_3 = DecoderBlockV7(3, 64, 32, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-5], edges[-3], target=y)

        return y


class DecoderBlockV7_(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 attn_type=None,
                 is_lastblock=False):
        super(DecoderBlockV7_, self).__init__()
        self.block_sn = block_sn
        self.attn_type = attn_type
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

        # rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        h, w = edge.shape[-2], edge.shape[-1]
        rgb_feature = F.interpolate(rgb_feature, size=(h, w), mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.attn_type is not None:
            x = self.attn(x)

        x = x + edge

        if self.block_sn == 3:
            x = F.interpolate(x, scale_factor=4.0, mode='bilinear')
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')

        x = self.conv2(x)

        return x


class DecoderV7_(nn.Module):
    def __init__(self, attn_type=None):
        super(DecoderV7_, self).__init__()

        self.decoder_block_1 = DecoderBlockV7_(1, 256, 128, output_feats=64, attn_type=attn_type)
        self.decoder_block_2 = DecoderBlockV7_(2, 128, 64, output_feats=32, target_feats=64, attn_type=attn_type)
        self.decoder_block_3 = DecoderBlockV7_(3, 64, 32, output_feats=1, target_feats=32, attn_type=attn_type,
                                              is_lastblock=True)

    def forward(self, features, edges, x=None):

        # print('features[-1].shape:', features[-1].shape)
        # print('features[-3].shape:', features[-3].shape)
        # print('features[-5].shape:', features[-5].shape)
        #
        # print('edges[-1].shape:', edges[-1].shape)
        # print('edges[-2].shape:', edges[-2].shape)
        # print('edges[-3].shape:', edges[-3].shape)

        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-5], edges[-3], target=y)

        return y


class DecoderBlockV7M_(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 attn_type=None,
                 scale_factor=4.0,
                 is_lastblock=False):
        super(DecoderBlockV7M_, self).__init__()
        self.block_sn = block_sn
        self.attn_type = attn_type
        self.is_lastblock = is_lastblock
        self.scale_factor = scale_factor

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
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
        # rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')
        rgb_feature = F.interpolate(rgb_feature, size=(edge.shape[-2], edge.shape[-1]), mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.attn_type is not None:
            x = self.attn(x)

        x = x + edge

        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')

        x = self.conv2(x)

        return x


class DecoderV7M_(nn.Module):
    def __init__(self, attn_type=None):
        super(DecoderV7M_, self).__init__()

        self.decoder_block_1 = DecoderBlockV7M_(1, 256, 128, output_feats=64, attn_type=attn_type, scale_factor=2.0)
        self.decoder_block_2 = DecoderBlockV7M_(2, 128, 64, output_feats=32, target_feats=64, attn_type=attn_type,
                                                scale_factor=2.0)
        self.decoder_block_3 = DecoderBlockV7M_(3, 64, 32, output_feats=1, target_feats=32, attn_type=attn_type,
                                                scale_factor=4.0, is_lastblock=True)

    def forward(self, features, edges, x=None):
        # for i in range(len(edges)):
        #     print('edges[{}].shape:{}'.format(i, edges[i].shape))

        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-5], edges[-3], target=y)

        return y


class DecoderV7N_(nn.Module):
    def __init__(self, attn_type=None):
        super(DecoderV7N_, self).__init__()

        self.decoder_block_1 = DecoderBlockV7_(1, 128, 64, output_feats=32, attn_type=attn_type)
        self.decoder_block_2 = DecoderBlockV7_(2, 64, 32, output_feats=16, target_feats=32, attn_type=attn_type)
        self.decoder_block_3 = DecoderBlockV7_(3, 32, 16, output_feats=1, target_feats=16, attn_type=attn_type,
                                              is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-5], edges[-3], target=y)

        return y


class DecoderBlockV7ECA(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_eca=True,
                 is_lastblock=False):
        super(DecoderBlockV7ECA, self).__init__()
        self.block_sn = block_sn
        self.use_eca = use_eca
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

        if use_eca is True:
            self.eca = eca_layer(edge_feats)
            # self.cbam = CBAM(edge_feats)

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
        rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.use_eca is True:
            # t0 = time.time()
            x = self.eca(x)
            # t1 = time.time() - t0
            # print('eca t1:{:.6f}s'.format(t1))

            # t0 = time.time()
            # x = self.cbam(x)
            # t2 = time.time() - t0
            # print('cbam t2:{:.6f}s'.format(t2))

        x = x + edge

        if self.block_sn == 3:
            x = F.interpolate(x, scale_factor=4.0, mode='bilinear')
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')

        x = self.conv2(x)

        return x


class DecoderV7ECA(nn.Module):
    def __init__(self):
        super(DecoderV7ECA, self).__init__()

        self.decoder_block_1 = DecoderBlockV7ECA(1, 256, 128, output_feats=64)
        self.decoder_block_2 = DecoderBlockV7ECA(2, 128, 64, output_feats=32, target_feats=64)
        self.decoder_block_3 = DecoderBlockV7ECA(3, 64, 32, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-5], edges[-3], target=y)

        return y


class DecoderBlockV7QnA(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_qna=True,
                 is_lastblock=False):
        super(DecoderBlockV7QnA, self).__init__()
        self.block_sn = block_sn
        self.use_qna = use_qna
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

        # if use_qna is True:
        #     self.eca = eca_layer(edge_feats)
        #     # self.cbam = CBAM(edge_feats)
        #     self.qna = QnAStage(stage=stage,
        #                         features=dim,
        #                         downsample=False,
        #                  config=self.config,
        #                  drop_path=drop_path[
        #                            drop_path_offset:drop_path_offset + qna_n_layers + int(
        #                                self.config.drop_path_downsample)],
        #                  **common_params
        #     )

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
        rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.use_eca is True:
            # t0 = time.time()
            x = self.eca(x)
            # t1 = time.time() - t0
            # print('eca t1:{:.6f}s'.format(t1))

            # t0 = time.time()
            # x = self.cbam(x)
            # t2 = time.time() - t0
            # print('cbam t2:{:.6f}s'.format(t2))

        x = x + edge

        if self.block_sn == 3:
            x = F.interpolate(x, scale_factor=4.0, mode='bilinear')
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')

        x = self.conv2(x)

        return x


class DecoderV7QnA(nn.Module):
    def __init__(self):
        super(DecoderV7QnA, self).__init__()

        self.decoder_block_1 = DecoderBlockV7QnA(1, 256, 128, output_feats=64)
        self.decoder_block_2 = DecoderBlockV7QnA(2, 128, 64, output_feats=32, target_feats=64)
        self.decoder_block_3 = DecoderBlockV7QnA(3, 64, 32, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-5], edges[-3], target=y)

        return y


class DecoderBlockV7SRM(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 use_srm=True,
                 is_lastblock=False):
        super(DecoderBlockV7SRM, self).__init__()
        self.block_sn = block_sn
        self.use_srm = use_srm
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

        if use_srm is True:
            self.cbam = CBAM(edge_feats)
            self.se = SELayer(edge_feats)
            self.eca = eca_layer(edge_feats)
            self.srm = SRMLayer(edge_feats)
            self.gct = GCTLayer(edge_feats)

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
        rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv1(x)

        if self.use_srm is True:
            # t0 = time.time()
            # x = self.cbam(x)
            # t1 = time.time() - t0
            # print('cbam:{:.6f}s'.format(t1))

            # t0 = time.time()
            # x = self.se(x)
            # t1 = time.time() - t0
            # print('se:{:.6f}s'.format(t1))

            # t0 = time.time()
            # x = self.eca(x)
            # t1 = time.time() - t0
            # print('eca:{:.6f}s'.format(t1))

            t0 = time.time()
            x = self.srm(x)
            t1 = time.time() - t0
            print('srm:{:.6f}s'.format(t1))

            # t0 = time.time()
            # x = self.gct(x)
            # t1 = time.time() - t0
            # print('gct:{:.6f}s'.format(t1))

        x = x + edge

        if self.block_sn == 3:
            x = F.interpolate(x, scale_factor=4.0, mode='bilinear')
        else:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear')

        x = self.conv2(x)

        return x


class DecoderV7SRM(nn.Module):
    def __init__(self):
        super(DecoderV7SRM, self).__init__()

        self.decoder_block_1 = DecoderBlockV7SRM(1, 256, 128, output_feats=64)
        self.decoder_block_2 = DecoderBlockV7SRM(2, 128, 64, output_feats=32, target_feats=64)
        self.decoder_block_3 = DecoderBlockV7SRM(3, 64, 32, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-5], edges[-3], target=y)

        return y


class DecoderBlockV8(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 is_lastblock=False,
                 use_cbam=False,
                 use_SE=True,
                 cbam_reduction_ratio=16):
        super(DecoderBlockV8, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam
        self.use_SE = use_SE
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
            self.cbam = CBAM(edge_feats, reduction_ratio=cbam_reduction_ratio)

        if use_SE is True:
            self.SE_block = SELayer(edge_feats, reduction=1)

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
        t0 = time.time()
        rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')
        t1 = time.time() - t0
        # print("F.interpolate time:{:.6f}s".format(t1))

        t0 = time.time()
        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)
        t2 = time.time() - t0
        # print("torch.cat time:{:.6f}s".format(t2))

        t0 = time.time()
        x = self.conv1(x)
        t3 = time.time() - t0
        # print("conv1 time:{:.6f}s".format(t3))

        t0 = time.time()
        if self.use_cbam is True:
            x = self.cbam(x)
        if self.use_SE is True:
            x = self.SE_block(x)
        t4 = time.time() - t0
        # print("attention mechanism time:{:.6f}s".format(t4))

        t0 = time.time()
        x = x + edge
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        t5 = time.time() - t0
        # print("add & F.interpolate time:{:.6f}s".format(t5))

        t0 = time.time()
        x = self.conv2(x)
        t6 = time.time() - t0
        # print("conv2 time:{:.6f}s".format(t6))

        return x


class DecoderV8(nn.Module):
    def __init__(self):
        super(DecoderV8, self).__init__()

        self.decoder_block_1 = DecoderBlockV8(1, 128, 128, output_feats=64)
        self.decoder_block_2 = DecoderBlockV8(2, 64, 64, output_feats=32, target_feats=64)
        self.decoder_block_3 = DecoderBlockV8(3, 32, 32, output_feats=16, target_feats=32)
        self.decoder_block_4 = DecoderBlockV8(4, 16, 16, output_feats=1, target_feats=16, is_lastblock=True)

        # self.decoder_block_1 = DecoderBlockV8(1, 128, 64, output_feats=32)
        # self.decoder_block_2 = DecoderBlockV8(2, 64, 32, output_feats=16, target_feats=32)
        # self.decoder_block_3 = DecoderBlockV8(3, 32, 16, output_feats=8, target_feats=16)
        # self.decoder_block_4 = DecoderBlockV8(4, 16, 8, output_feats=1, target_feats=8, is_lastblock=True,
        #                                       cbam_reduction_ratio=8)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-5], edges[-3], target=y)
        y = self.decoder_block_4(features[-7], edges[-4], target=y)

        return y


class DecoderBlockV9(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 is_lastblock=False,
                 use_cbam=False,
                 use_SE=True,
                 cbam_reduction_ratio=16):
        super(DecoderBlockV9, self).__init__()
        self.block_sn = block_sn
        self.use_cbam = use_cbam
        self.use_SE = use_SE
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
            self.cbam = CBAM(edge_feats, reduction_ratio=cbam_reduction_ratio)

        if use_SE is True:
            self.SE_block = SELayer(edge_feats, reduction=1)

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
        if self.is_lastblock is True:
            t0 = time.time()
            rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')
            t1 = time.time() - t0
            # print("F.interpolate time:{:.6f}s".format(t1))

        t0 = time.time()
        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)
        t2 = time.time() - t0
        # print("torch.cat time:{:.6f}s".format(t2))

        t0 = time.time()
        x = self.conv1(x)
        t3 = time.time() - t0
        # print("conv1 time:{:.6f}s".format(t3))

        t0 = time.time()
        if self.use_cbam is True:
            x = self.cbam(x)
        if self.use_SE is True:
            x = self.SE_block(x)
        t4 = time.time() - t0
        # print("attention mechanism time:{:.6f}s".format(t4))

        t0 = time.time()
        x = x + edge
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        t5 = time.time() - t0
        # print("add & F.interpolate time:{:.6f}s".format(t5))

        t0 = time.time()
        x = self.conv2(x)
        t6 = time.time() - t0
        # print("conv2 time:{:.6f}s".format(t6))

        return x


class DecoderV9(nn.Module):
    def __init__(self):
        super(DecoderV9, self).__init__()

        self.decoder_block_1 = DecoderBlockV9(1, 128, 128, output_feats=64)
        self.decoder_block_2 = DecoderBlockV9(2, 64, 64, output_feats=32, target_feats=64)
        self.decoder_block_3 = DecoderBlockV9(3, 32, 32, output_feats=16, target_feats=32)
        self.decoder_block_4 = DecoderBlockV9(4, 16, 16, output_feats=8, target_feats=16)
        self.decoder_block_5 = DecoderBlockV9(5, 16, 8, output_feats=1, target_feats=8, is_lastblock=True)

    def forward(self, features, edges, x=None):
        # for i in range(7):
        #     print('features:{} edges:{}'.format(features[i].shape, edges[i].shape))
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-3], edges[-2], target=y)
        y = self.decoder_block_3(features[-5], edges[-3], target=y)
        y = self.decoder_block_4(features[-7], edges[-4], target=y)
        y = self.decoder_block_5(features[-7], edges[-5], target=y)

        return y


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':  # default
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):
        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):
    # patch_size = 4, patch_stride = 4, in_chans = 3, embed_dim = 96
    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        # input:[b, 3, h, w]  output:[b, embed_dim, h/4, w/4]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)  # nn.BatchNorm2d(num_features=embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        # input:[b, c, h, w]  output:[b, 2*c, h/2, w/2]
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


res_dict = {
    'nyu_reduced_full': (480, 640),
    'nyu_reduced_half': (240, 320),
    'nyu_reduced_mini': (224, 224),
    'kitti_full': (384, 1280),
    'kitti_half': (192, 640)
}


Embedding_Channels = {'T0': 40, 'T1': 64, 'T2': 96, 'S': 128,
                      'X': 128, 'X-V2': 128, 'X-V3': 128}
depth_dicts = {'T0': (1, 2, 8, 2),
               'T1': (1, 2, 8, 2),
               'T2': (1, 2, 8, 2),
               'S': (1, 2, 13, 2),
               'X': (1, 2), 'X-V2': (1, 2, 13), 'X-V3': (1, 2, 8, 1)}


class FasterNetEdge(nn.Module):
    def __init__(self,
                 opts,
                 in_chans=3,
                 num_classes=1000,
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 fork_feat=False,
                 pconv_fw_type='split_cat'):
        super(FasterNetEdge, self).__init__()

        dataset_name = getattr(opts, 'dataset.name', 'nyu_reduced')
        res_mode = getattr(opts, 'common.resolution', 'full')
        res_key = '{}_{}'.format(dataset_name, res_mode)
        self.resolution = res_dict[res_key]

        bs = getattr(opts, "common.bs", 8)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        mode = getattr(opts, "model.mode", 'X')
        embed_dim = Embedding_Channels[mode]
        depths = depth_dicts[mode]
        logger.info("embed_dim in class FasterNetPCGUB:{}".format(embed_dim))

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)  # 4
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))  # 96 * 8
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # NOTE: Encoder Params(X:0.81M, X-V2:M)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** i_stage),
                                 norm_layer=norm_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        # NOTE: Encoder Params(X:0.81M, X-V2:16.91M, X-V3:M)
        # [B, 128, 120, 160] stage1 !
        # [B, 256, 60, 80] merging
        # [B, 256, 60, 80] stage2   !
        # [B, 512, 30, 40] merging
        # [B, 512, 30, 40] stage3   !
        # [B, 1024, 15, 20] merging
        # [B, 1024, 15, 20] stage4  !
        # ====== T0:
        # [B, 40, 120, 160] stage1 !
        # [B, 80, 60, 80] merging
        # [B, 80, 60, 80] stage2   !
        # [B, 160, 30, 40] merging
        # [B, 160, 30, 40] stage3   !
        # [B, 320, 15, 20] merging
        # [B, 320, 15, 20] stage4  !

        self.btlnck_conv1 = nn.Conv2d(320, 160, kernel_size=1, stride=1, padding=0)  # btlnck conv
        self.btlnck_conv2 = nn.Conv2d(160, 128, kernel_size=1, stride=1, padding=0)  # btlnck conv
        self.btlnck_conv3 = nn.Conv2d(80, 64, kernel_size=1, stride=1, padding=0)  # btlnck conv
        self.btlnck_conv4 = nn.Conv2d(40, 32, kernel_size=1, stride=1, padding=0)

        self.edge_feature_extractor = ENV4(reduction=1)
        self.transition_module = TransitionModuleV3M4(feats_list=[64, 128, 160], alpha='learnable')
        self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

    def rgb_feature_extractor(self, x):
        # NOTE: 提取RGB的contextual特征信息
        x = self.patch_embed(x)  # [B, embed_dim, h/4, w/4]

        rgb_features = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            rgb_features.append(x)

        rgb_features_ = []
        rgb_features_.append(self.btlnck_conv1(rgb_features[-1]))
        rgb_features_.append(self.btlnck_conv2(rgb_features[-3]))
        rgb_features_.append(self.btlnck_conv3(rgb_features[-5]))
        if self.btlnck_conv4 is not None:
            rgb_features_.append(self.btlnck_conv4(rgb_features[-7]))

        return rgb_features_

    def forward(self, x, speed_test=False):
        # t0 = time.time()
        rgb_features = self.rgb_feature_extractor(x)  # NOTE: 提取RGB的特征信息
        # t1 = time.time() - t0
        # for i in range(len(rgb_features)):
        #     print("{}, rgb_feature.shape:{}".format(i, rgb_features[i].shape))
        # exit()
        rgb_features_ = rgb_features[:3]
        x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
        edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
        fused_features = self.transition_module(rgb_features_, edge_features)  # NOTE: 融合RGB和edge信息过渡给decoder
        fused_features.append(rgb_features[-1])  # 把1/4尺度的特征图添加进去
        y = self.decoder(fused_features)

        return y  # y

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def edge_extractor_torch(self, x: Tensor, device="cuda:0"):
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
        # sobel_xy = normalize2img_tensor(sobel_xy)
        sobel_xy = self.normalize_to_01_torch(sobel_xy)

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


# NOTE: 最近的一版
# class FasterNetEdge(nn.Module):
#     def __init__(self,
#                  opts,
#                  in_chans=3,
#                  num_classes=1000,
#                  mlp_ratio=2.,
#                  n_div=4,
#                  patch_size=4,
#                  patch_stride=4,
#                  patch_size2=2,  # for subsequent layers
#                  patch_stride2=2,
#                  patch_norm=True,
#                  feature_dim=1280,
#                  drop_path_rate=0.1,
#                  layer_scale_init_value=0,
#                  norm_layer='BN',
#                  act_layer='RELU',
#                  fork_feat=False,
#                  pconv_fw_type='split_cat'):
#         super(FasterNetEdge, self).__init__()
#
#         dataset_name = getattr(opts, 'dataset.name', 'nyu_reduced')
#         res_mode = getattr(opts, 'common.resolution', 'full')
#         res_key = '{}_{}'.format(dataset_name, res_mode)
#         self.resolution = res_dict[res_key]
#
#         bs = getattr(opts, "common.bs", 8)
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#         mode = getattr(opts, "model.mode", 'X')
#         embed_dim = Embedding_Channels[mode]
#         depths = depth_dicts[mode]
#         logger.info("embed_dim in class FasterNetPCGUB:{}".format(embed_dim))
#
#         if norm_layer == 'BN':
#             norm_layer = nn.BatchNorm2d
#         else:
#             raise NotImplementedError
#
#         if act_layer == 'GELU':
#             act_layer = nn.GELU
#         elif act_layer == 'RELU':
#             act_layer = partial(nn.ReLU, inplace=True)
#         else:
#             raise NotImplementedError
#
#         if not fork_feat:
#             self.num_classes = num_classes
#         self.num_stages = len(depths)  # 4
#         self.embed_dim = embed_dim
#         self.patch_norm = patch_norm
#         self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))  # 96 * 8
#         self.mlp_ratio = mlp_ratio
#         self.depths = depths
#
#         # NOTE: Encoder Params(X:0.81M, X-V2:M)
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size,
#             patch_stride=patch_stride,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None
#         )
#
#         # stochastic depth decay rule
#         dpr = [x.item()
#                for x in torch.linspace(0, drop_path_rate, sum(depths))]
#
#         # build layers
#         stages_list = []
#         for i_stage in range(self.num_stages):
#             stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
#                                n_div=n_div,
#                                depth=depths[i_stage],
#                                mlp_ratio=self.mlp_ratio,
#                                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
#                                layer_scale_init_value=layer_scale_init_value,
#                                norm_layer=norm_layer,
#                                act_layer=act_layer,
#                                pconv_fw_type=pconv_fw_type
#                                )
#             stages_list.append(stage)
#
#             # patch merging layer
#             if i_stage < self.num_stages - 1:
#                 stages_list.append(
#                     PatchMerging(patch_size2=patch_size2,
#                                  patch_stride2=patch_stride2,
#                                  dim=int(embed_dim * 2 ** i_stage),
#                                  norm_layer=norm_layer)
#                 )
#
#         self.stages = nn.Sequential(*stages_list)
#
#         # NOTE: Encoder Params(X:0.81M, X-V2:16.91M, X-V3:M)
#         # [B, 128, 120, 160] stage1 !
#         # [B, 256, 60, 80] merging
#         # [B, 256, 60, 80] stage2   !
#         # [B, 512, 30, 40] merging
#         # [B, 512, 30, 40] stage3   !
#         # [B, 1024, 15, 20] merging
#         # [B, 1024, 15, 20] stage4  !
#
#         # NOTE: botleneck convolution for EdgeNetV2+DecoderV7
#         reduction_ratio = 4
#         self.btlnck_conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         self.btlnck_conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         self.btlnck_conv3 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         self.btlnck_conv4 = None
#
#         # NOTE: botleneck convolution for EdgeNetV2N+DecoderV7N
#         # reduction_ratio = 8
#         # self.btlnck_conv1 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv3 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv4 = None
#
#         # NOTE: botleneck convolution for EdgeNetV3+DecoderV8 EdgeNetV4+DecoderV9
#         # self.btlnck_conv1 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv3 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv4 = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0)  # btlnck conv
#
#         # NOTE: EdgeNet
#         # self.edge_feature_extractor = EdgeNetV1()  # NOTE: Params(0.26M)
#         # [B, 32, 240, 320]  !
#         # [B, 64, 120, 160]  !
#         # [B, 128, 60, 80]   !
#         # [B, 256, 30, 40]   !
#         # self.edge_feature_extractor = EdgeNetV1_4L()  # NOTE: Params(0.88M)
#         # self.edge_feature_extractor = EdgeNetV1_4L_N()  # NOTE: Params(0.88M)
#         # [B, 32, 120, 160]  !
#         # [B, 64, 60, 80]    !
#         # [B, 128, 30, 40]   !
#         # self.edge_feature_extractor = EdgeNetV2()  # NOTE: Params(0.22M)
#         # [B, 32, 240, 320]  !
#         # [B, 64, 120, 160]  !
#         # [B, 128, 60, 80]   !
#         # [B, 256, 30, 40]   !
#         # self.edge_feature_extractor = EdgeNetV3()  # NOTE: Params(0.88M)
#         # self.edge_feature_extractor = EdgeNetV4()  # NOTE: Params(0.88M)
#         # self.edge_feature_extractor = EdgeNetV2N()  # NOTE: Params(0.22M)
#         # self.edge_feature_extractor = EdgeNetV2Attn(attn_type='cbams')
#         # self.edge_feature_extractor = EdgeNetV5(resolution=self.resolution)
#         self.edge_feature_extractor = EdgeNetV2TF(img_shape=(480, 640))  # NOTE: last
#
#         # NOTE: Decoder
#         # [B, 16, 240, 320]  !
#         # [B, 64, 120, 160]  !
#         # [B, 64, 60, 80]    !
#         # self.decoder = DecoderV6()  # NOTE: Params(0.45M)
#         # [B, 32, 240, 320]  !
#         # [B, 64, 120, 160]  !
#         # [B, 128, 60, 80]   !
#         # [B, 256, 30, 40]   !
#         # self.decoder = DecoderV6_4L()  # NOTE: Params(3.6M)
#         # self.decoder = DecoderV6_4L_N()  # NOTE: Params(M)
#         # self.decoder = DecoderV6_4L_N_woRGBFeats()  # NOTE: Params(M)
#         # self.decoder = DecoderV6_4L_woRGBFeats()
#         # self.decoder = DecoderV7()  # NOTE: Params(0.73M)
#         # self.decoder = DecoderV8()  # NOTE: Params(2.16M) 31.98 29.82 28.94
#         # self.decoder = DecoderV9()  # NOTE: Params(2.16M)
#         # self.decoder = DecoderV7ECA()
#         # self.decoder = DecoderV7SRM()
#         # self.decoder = DecoderV7_(attn_type='se')
#         # self.decoder = DecoderV7N_(attn_type='cbam')
#         # self.decoder = DecoderV7_(attn_type='cbamc')
#         # self.decoder = DecoderV7_(attn_type='cbam')
#         self.decoder = DecoderV7M_(attn_type='cbam')  # NOTE: last
#
#         # # NOTE: TopFormer
#         # self.local_token_generator = LTG()  # NOTE: Params(0.4M)
#         # self.global_token_generator = GTG()  # NOTE: Non-Parametric Operation
#         #
#         # norm_cfg = dict(type='BN', requires_grad=True)
#         # act_layer = nn.ReLU6
#         # self.local_token_channels = [128, 64, 32]
#         # dpr = [x.item() for x in torch.linspace(0, 0.1, 4)]
#         # # NOTE: Params(1.52M)  29.38M  30.90M
#         # self.transformer_layer = BasicLayer(
#         #     block_num=4,  # 4
#         #     embedding_dim=sum(self.local_token_channels),  # 384
#         #     key_dim=16,  # 16
#         #     num_heads=8,  # 8
#         #     mlp_ratio=2,  # 2
#         #     attn_ratio=2,  # 2
#         #     drop=0, attn_drop=0,
#         #     drop_path=dpr,  # [0.025, 0.050, 0.075, 0.100]
#         #     norm_cfg=norm_cfg,  # dict(type='SyncBN', requires_grad=True)
#         #     act_layer=act_layer)  # nn.ReLU6
#         #
#         # # Semantic Injection Module
#         # self.out_channels = [128, 128, 128]  # local token enhanced by global token
#         # # NOTE: Params(0.09M)  30.90M  30.99M
#         # self.sim_list = nn.ModuleList()
#         # for i in range(len(self.local_token_channels)):
#         #     self.sim_list.append(SemanticInjectionModule(self.local_token_channels[i], self.out_channels[i], norm_cfg=norm_cfg, activations=act_layer))
#         #
#         # # NOTE: Params(0.04M)  30.99M  31.03M
#         # self.dense_head = RRDecoderV1()
#
#     def rgb_feature_extractor(self, x):
#         # NOTE: 提取RGB的contextual特征信息
#         x = self.patch_embed(x)  # [B, embed_dim, h/4, w/4]
#         # print("x.shape:\n", x.shape)
#
#         rgb_features = []
#         for idx, stage in enumerate(self.stages):
#             x = stage(x)
#             # print("{}, x.shape:{}".format(idx, x.shape))
#             rgb_features.append(x)
#
#         rgb_features[-1] = self.btlnck_conv1(rgb_features[-1])
#         rgb_features[-3] = self.btlnck_conv2(rgb_features[-3])
#         rgb_features[-5] = self.btlnck_conv3(rgb_features[-5])
#         if self.btlnck_conv4 is not None:
#             rgb_features[-7] = self.btlnck_conv4(rgb_features[-7])
#
#         return rgb_features
#
#     def forward(self, x):
#         # NOTE: 提取RGB的特征信息
#         # t0 = time.time()
#         rgb_features = self.rgb_feature_extractor(x)
#         # t1 = time.time() - t0
#         # print("rgb_feature_extractor time:{:.6f}s".format(time.time() - t0))
#         # print("rgb_feature_extractor time:{:.4f}s".format(t1))
#         # for i in range(len(rgb_features)):
#         #     print("{}, rgb_feature.shape:{}".format(i, rgb_features[i].shape))
#         # exit()
#
#         # NOTE: 提取RGB的edge
#         # t0 = time.time()
#         x_edge = self.edge_extractor_torch(x, device=self.device)
#         # t2 = time.time() - t0
#         # print("edge_extractor_torch time:{:.6f}s".format(time.time() - t0))
#
#         # NOTE: 提取edge的特征信息
#         # t0 = time.time()
#         # edge_features, temp1, temp2 = self.edge_feature_extractor(x_edge)
#         edge_features = self.edge_feature_extractor(x_edge)
#         # t3 = time.time() - t0
#         # print("edge_feature_extractor time:{:.6f}s".format(time.time() - t0))
#         # for i in range(len(edge_features)):
#         #     print("{}, edge_features.shape:{}".format(i, edge_features[i].shape))
#         # exit()
#
#         # NOTE: 上采样稠密深度估计
#         # t0 = time.time()
#         y = self.decoder(rgb_features, edge_features, x)
#         # t4 = time.time() - t0
#         # for i in range(len(res)):
#         #     print("decoder_{}:{}".format(i, res[i].shape))
#         # print("decoder time:{:.4f}s".format(t2 + t3 + t4))
#         # exit()
#
#         # # NOTE: TopFormer结构操作
#         # # [b, 128, 30, 40]
#         # # [b, 64, 60, 80]
#         # # [b, 32, 120, 160]
#         # local_tokens = self.local_token_generator(rgb_features, edge_features)
#         # # for i in range(len(local_tokens)):
#         # #     print('local_tokens[{}].shape:{}'.format(i, local_tokens[i].shape))
#         #
#         # global_token = self.global_token_generator(local_tokens)
#         # # print('global_token.shape:', global_token.shape)
#         #
#         # global_token = self.transformer_layer(global_token)  # [b, 224, 30, 40]
#         #
#         # # [b, 128, 30, 40]
#         # # [b, 64, 30, 40]
#         # # [b, 32, 30, 40]
#         # xx = global_token.split(self.local_token_channels, dim=1)
#         #
#         # inject_tokens = []
#         # for i in range(len(self.local_token_channels)):
#         #     local_token = local_tokens[i]
#         #     global_semantics = xx[i]
#         #     out_ = self.sim_list[i](local_token, global_semantics)
#         #     inject_tokens.append(out_)
#         # # for i in range(len(inject_tokens)):
#         # #     print('inject_tokens[{}].shape:{}'.format(i, inject_tokens[i].shape))
#         #
#         # y = self.dense_head(inject_tokens)
#
#         return y  # y inject_tokens rgb_features   temp1, temp2
#
#     def cls_init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def edge_extractor_torch(self, x: Tensor, device="cuda:0"):
#         conv_rgb_core_sobel_horizontal = [
#             [[1, 2, 1], [0, 0, 0], [-1, -2, -1],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0]
#              ],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [1, 2, 1], [0, 0, 0], [-1, -2, -1],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0]
#              ],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [1, 2, 1], [0, 0, 0], [-1, -2, -1],
#              ]]
#         conv_rgb_core_sobel_vertical = [
#             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0]
#              ],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0]
#              ],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
#              ]]
#
#         conv_op_horizontal = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
#         conv_op_vertical = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
#
#         sobel_kernel_horizontal = np.array(conv_rgb_core_sobel_horizontal, dtype='float32')
#         sobel_kernel_horizontal = sobel_kernel_horizontal.reshape((3, 3, 3, 3))
#         conv_op_horizontal.weight.data = torch.from_numpy(sobel_kernel_horizontal).to(device)
#
#         sobel_kernel_vertical = np.array(conv_rgb_core_sobel_vertical, dtype='float32')
#         sobel_kernel_vertical = sobel_kernel_vertical.reshape((3, 3, 3, 3))
#         conv_op_vertical.weight.data = torch.from_numpy(sobel_kernel_vertical).to(device)
#
#         sobel_x = conv_op_horizontal(x)
#         sobel_y = conv_op_vertical(x)
#         sobel_xy = torch.sqrt(torch.square(sobel_x) + torch.square(sobel_y))
#
#         # sobel_x = normalize2img_tensor(sobel_x)
#         # sobel_y = normalize2img_tensor(sobel_y)
#         sobel_xy = normalize2img_tensor(sobel_xy)
#
#         return sobel_xy  # sobel_x, sobel_y, sobel_xy


# class FasterNetEdge(nn.Module):
#     def __init__(self,
#                  opts,
#                  in_chans=3,
#                  num_classes=1000,
#                  mlp_ratio=2.,
#                  n_div=4,
#                  patch_size=4,
#                  patch_stride=4,
#                  patch_size2=2,  # for subsequent layers
#                  patch_stride2=2,
#                  patch_norm=True,
#                  feature_dim=1280,
#                  drop_path_rate=0.1,
#                  layer_scale_init_value=0,
#                  norm_layer='BN',
#                  act_layer='RELU',
#                  fork_feat=False,
#                  pconv_fw_type='split_cat'):
#         super(FasterNetEdge, self).__init__()
#
#         dataset_name = getattr(opts, 'dataset.name', 'nyu_reduced')
#         res_mode = getattr(opts, 'common.resolution', 'full')
#         res_key = '{}_{}'.format(dataset_name, res_mode)
#         self.resolution = res_dict[res_key]
#
#         bs = getattr(opts, "common.bs", 8)
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#         mode = getattr(opts, "model.mode", 'X')
#         embed_dim = Embedding_Channels[mode]
#         depths = depth_dicts[mode]
#         logger.info("embed_dim in class FasterNetPCGUB:{}".format(embed_dim))
#
#         if norm_layer == 'BN':
#             norm_layer = nn.BatchNorm2d
#         else:
#             raise NotImplementedError
#
#         if act_layer == 'GELU':
#             act_layer = nn.GELU
#         elif act_layer == 'RELU':
#             act_layer = partial(nn.ReLU, inplace=True)
#         else:
#             raise NotImplementedError
#
#         if not fork_feat:
#             self.num_classes = num_classes
#         self.num_stages = len(depths)  # 4
#         self.embed_dim = embed_dim
#         self.patch_norm = patch_norm
#         self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))  # 96 * 8
#         self.mlp_ratio = mlp_ratio
#         self.depths = depths
#
#         # NOTE: Encoder Params(X:0.81M, X-V2:M)
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size,
#             patch_stride=patch_stride,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None
#         )
#
#         # stochastic depth decay rule
#         dpr = [x.item()
#                for x in torch.linspace(0, drop_path_rate, sum(depths))]
#
#         # build layers
#         stages_list = []
#         for i_stage in range(self.num_stages):
#             stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
#                                n_div=n_div,
#                                depth=depths[i_stage],
#                                mlp_ratio=self.mlp_ratio,
#                                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
#                                layer_scale_init_value=layer_scale_init_value,
#                                norm_layer=norm_layer,
#                                act_layer=act_layer,
#                                pconv_fw_type=pconv_fw_type
#                                )
#             stages_list.append(stage)
#
#             # patch merging layer
#             if i_stage < self.num_stages - 1:
#                 stages_list.append(
#                     PatchMerging(patch_size2=patch_size2,
#                                  patch_stride2=patch_stride2,
#                                  dim=int(embed_dim * 2 ** i_stage),
#                                  norm_layer=norm_layer)
#                 )
#
#         self.stages = nn.Sequential(*stages_list)
#
#         # NOTE: Encoder Params(X:0.81M, X-V2:16.91M, X-V3:M)
#         # [B, 128, 120, 160] stage1 !
#         # [B, 256, 60, 80] merging
#         # [B, 256, 60, 80] stage2   !
#         # [B, 512, 30, 40] merging
#         # [B, 512, 30, 40] stage3   !
#         # [B, 1024, 15, 20] merging
#         # [B, 1024, 15, 20] stage4  !
#
#         # NOTE: botleneck convolution for EdgeNetV2+DecoderV7
#         reduction_ratio = 4
#         self.btlnck_conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         self.btlnck_conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         self.btlnck_conv3 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         self.btlnck_conv4 = None
#
#         # NOTE: botleneck convolution for EdgeNetV2N+DecoderV7N
#         # reduction_ratio = 8
#         # self.btlnck_conv1 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv3 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv4 = None
#
#         # NOTE: botleneck convolution for EdgeNetV3+DecoderV8 EdgeNetV4+DecoderV9
#         # self.btlnck_conv1 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv3 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)  # btlnck conv
#         # self.btlnck_conv4 = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0)  # btlnck conv
#
#         # NOTE: EdgeNet
#         # self.edge_feature_extractor = EdgeNetV1()  # NOTE: Params(0.26M)
#         # [B, 32, 240, 320]  !
#         # [B, 64, 120, 160]  !
#         # [B, 128, 60, 80]   !
#         # [B, 256, 30, 40]   !
#         # self.edge_feature_extractor = EdgeNetV1_4L()  # NOTE: Params(0.88M)
#         # self.edge_feature_extractor = EdgeNetV1_4L_N()  # NOTE: Params(0.88M)
#         # [B, 32, 120, 160]  !
#         # [B, 64, 60, 80]    !
#         # [B, 128, 30, 40]   !
#         self.edge_feature_extractor = EdgeNetV2()  # NOTE: Params(0.22M)
#         # [B, 32, 240, 320]  !
#         # [B, 64, 120, 160]  !
#         # [B, 128, 60, 80]   !
#         # [B, 256, 30, 40]   !
#         # self.edge_feature_extractor = EdgeNetV3()  # NOTE: Params(0.88M)
#         # self.edge_feature_extractor = EdgeNetV4()  # NOTE: Params(0.88M)
#         # self.edge_feature_extractor = EdgeNetV2N()  # NOTE: Params(0.22M)
#         # self.edge_feature_extractor = EdgeNetV2Attn(attn_type='cbams')
#         # self.edge_feature_extractor = EdgeNetV5(resolution=self.resolution)
#         # self.edge_feature_extractor = EdgeNetV2TF(img_shape=(480, 640))
#
#         # NOTE: Decoder
#         # [B, 16, 240, 320]  !
#         # [B, 64, 120, 160]  !
#         # [B, 64, 60, 80]    !
#         # self.decoder = DecoderV6()  # NOTE: Params(0.45M)
#         # [B, 32, 240, 320]  !
#         # [B, 64, 120, 160]  !
#         # [B, 128, 60, 80]   !
#         # [B, 256, 30, 40]   !
#         # self.decoder = DecoderV6_4L()  # NOTE: Params(3.6M)
#         # self.decoder = DecoderV6_4L_N()  # NOTE: Params(M)
#         # self.decoder = DecoderV6_4L_N_woRGBFeats()  # NOTE: Params(M)
#         # self.decoder = DecoderV6_4L_woRGBFeats()
#         self.decoder = DecoderV7()  # NOTE: Params(0.73M)
#         # self.decoder = DecoderV8()  # NOTE: Params(2.16M) 31.98 29.82 28.94
#         # self.decoder = DecoderV9()  # NOTE: Params(2.16M)
#         # self.decoder = DecoderV7ECA()
#         # self.decoder = DecoderV7SRM()
#         # self.decoder = DecoderV7_(attn_type='se')
#         # self.decoder = DecoderV7N_(attn_type='cbam')
#         # self.decoder = DecoderV7_(attn_type='cbamc')
#         # self.decoder = DecoderV7_(attn_type='cbam')
#
#     def rgb_feature_extractor(self, x):
#         # NOTE: 提取RGB的contextual特征信息
#         x = self.patch_embed(x)  # [B, embed_dim, h/4, w/4]
#         # print("x.shape:\n", x.shape)
#
#         rgb_features = []
#         for idx, stage in enumerate(self.stages):
#             x = stage(x)
#             # print("{}, x.shape:{}".format(idx, x.shape))
#             rgb_features.append(x)
#
#         rgb_features[-1] = self.btlnck_conv1(rgb_features[-1])
#         rgb_features[-3] = self.btlnck_conv2(rgb_features[-3])
#         rgb_features[-5] = self.btlnck_conv3(rgb_features[-5])
#         if self.btlnck_conv4 is not None:
#             rgb_features[-7] = self.btlnck_conv4(rgb_features[-7])
#
#         return rgb_features
#
#     def forward(self, x):
#         # NOTE: 提取RGB的特征信息
#         t0 = time.time()
#         rgb_features = self.rgb_feature_extractor(x)
#         t1 = time.time() - t0
#         # print("rgb_feature_extractor time:{:.4f}s".format(t1))
#         # for i in range(len(rgb_features)):
#         #     print("{}, rgb_feature.shape:{}".format(i, rgb_features[i].shape))
#         # exit()
#
#         # NOTE: 提取RGB的edge
#         t0 = time.time()
#         x_edge = self.edge_extractor_torch(x, device=self.device)
#         t2 = time.time() - t0
#         # print("edge_extractor_torch time:{:.6f}s".format(time.time() - t0))
#
#         # NOTE: 提取edge的特征信息
#         t0 = time.time()
#         edge_features = self.edge_feature_extractor(x_edge)
#         t3 = time.time() - t0
#         # print("edge_feature_extractor time:{:.6f}s".format(time.time() - t0))
#         # for i in range(len(edge_features)):
#         #     print("{}, edge_features.shape:{}".format(i, edge_features[i].shape))
#         # exit()
#
#         # NOTE: 上采样稠密深度估计
#         t0 = time.time()
#         y = self.decoder(rgb_features, edge_features, x)
#         t4 = time.time() - t0
#         # for i in range(len(res)):
#         #     print("decoder_{}:{}".format(i, res[i].shape))
#         # print("decoder time:{:.4f}s".format(t2 + t3 + t4))
#
#         return y  # y inject_tokens
#
#     def cls_init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def edge_extractor_torch(self, x: Tensor, device="cuda:0"):
#         conv_rgb_core_sobel_horizontal = [
#             [[1, 2, 1], [0, 0, 0], [-1, -2, -1],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0]
#              ],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [1, 2, 1], [0, 0, 0], [-1, -2, -1],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0]
#              ],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [1, 2, 1], [0, 0, 0], [-1, -2, -1],
#              ]]
#         conv_rgb_core_sobel_vertical = [
#             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0]
#              ],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0]
#              ],
#             [[0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [0, 0, 0], [0, 0, 0], [0, 0, 0],
#              [-1, 0, 1], [-2, 0, 2], [-1, 0, 1],
#              ]]
#
#         conv_op_horizontal = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
#         conv_op_vertical = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)
#
#         sobel_kernel_horizontal = np.array(conv_rgb_core_sobel_horizontal, dtype='float32')
#         sobel_kernel_horizontal = sobel_kernel_horizontal.reshape((3, 3, 3, 3))
#         conv_op_horizontal.weight.data = torch.from_numpy(sobel_kernel_horizontal).to(device)
#
#         sobel_kernel_vertical = np.array(conv_rgb_core_sobel_vertical, dtype='float32')
#         sobel_kernel_vertical = sobel_kernel_vertical.reshape((3, 3, 3, 3))
#         conv_op_vertical.weight.data = torch.from_numpy(sobel_kernel_vertical).to(device)
#
#         sobel_x = conv_op_horizontal(x)
#         sobel_y = conv_op_vertical(x)
#         sobel_xy = torch.sqrt(torch.square(sobel_x) + torch.square(sobel_y))
#
#         # sobel_x = normalize2img_tensor(sobel_x)
#         # sobel_y = normalize2img_tensor(sobel_y)
#         sobel_xy = normalize2img_tensor(sobel_xy)
#
#         return sobel_xy  # sobel_x, sobel_y, sobel_xy


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()

        # self.edge_feature_extractor = EdgeNetV2()
        self.edge_feature_extractor = EdgeNetV2TF()

    def forward(self, x):
        x = self.edge_feature_extractor(x)

        return x


def get_model_params(model):
    """
    计算模型参数量 单位MB
    Args:
        model: 网络模型
    Returns: None
    """
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: %.2fM' % (total / 1e6))


if __name__ == '__main__':
    model = TestNet()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    get_model_params(model)

    X = torch.randn(size=(4, 3, 480, 640)).to(device)
    res = model.forward(X)
    print('res.shape:', res.shape)
    exit()



