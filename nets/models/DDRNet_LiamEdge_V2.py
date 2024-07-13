import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import logger
from model.DDRNet_23_slim import DualResNet_Backbone
from ..modules.se import SELayer
from ..modules.cbam import CBAM
from ..modules.eca_module import eca_layer
from ..modules.srm_module import SRMLayer
from ..modules.gct_module import GCTLayer
from ..modules.tf_block_raw import TFPatchEmbed, MyAttention, TFDropPath, PatchEmbedUnfold, SimilarityQK, GetV
from ..modules.tf_block_liam import AttnBlock
from ..modules.tf_block_topformer import Attention, PyramidPoolAgg, BasicLayer, SemanticInjectionModule
from .MobileNetV2_LiamEdge import EdgeNetV4 as ENV4
from .MobileNetV2_LiamEdge import TransitionModuleV3M4, OutputHeadV4, SGNet
from ..modules.GLPDepth_decoder import SelectiveFeatureFusion
from ..modules.tf_block_topformer import CrossModalTFBlockV3M1, TFBlock, get_shape

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time


def normalize2img_tensor(x: Tensor):
    min_val = x.min()
    max_val = x.max()
    res = (x - min_val) / (max_val - min_val)
    res = res * 255.
    return res


class EdgeNetV1(nn.Module):
    def __init__(self, reduction=1, img_shape=(480, 640)):
        super(EdgeNetV1, self).__init__()
        self.reduction = reduction
        h, w = img_shape[0], img_shape[1]

        input_feats = 3
        output_feats = 32 // reduction
        # print('input_feats:{} output_feats:{}'.format(input_feats, output_feats))
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/4, w/4]

        self.drop_path_1 = TFDropPath(0.025)
        ps = 8
        h_ = h // 4
        w_ = w // 4
        conv_output_channels = output_feats
        # print('conv_output_channels:{}'.format(conv_output_channels))
        self.embed_1 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_1 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_1 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_1 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        input_feats = output_feats
        output_feats = 64 // reduction
        # print('input_feats:{} output_feats:{}'.format(input_feats, output_feats))
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]

        self.drop_path_2 = TFDropPath(0.050)
        ps = 4
        h_ = h // 8
        w_ = w // 8
        conv_output_channels = output_feats
        # print('conv_output_channels:{}'.format(conv_output_channels))
        self.embed_2 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_2 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_2 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_2 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        input_feats = output_feats
        output_feats = 128 // reduction
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/16, w/16]

        self.drop_path_3 = TFDropPath(0.075)
        ps = 2
        h_ = h // 16
        w_ = w // 16
        conv_output_channels = output_feats
        self.embed_3 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_3 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_3 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_3 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        input_feats = output_feats
        output_feats = 256 // reduction
        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 256, h/32, w/32]

        self.drop_path_4 = TFDropPath(0.100)
        ps = 1
        h_ = h // 32
        w_ = w // 32
        conv_output_channels = output_feats
        self.embed_4 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_4 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_4 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_4 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

    def forward(self, x):
        features = []
        x1 = self.edge_encoder_1(x)  # [B, 32, 120, 160]
        x2 = self.edge_encoder_2(x1)  # [B, 64, 60, 80]
        x3 = self.edge_encoder_3(x2)  # [B, 128, 30, 40]
        x4 = self.edge_encoder_4(x3)  # [B, 256, 15, 20]

        # print('x1.shape:{}'.format(x1.shape))
        # print('x2.shape:{}'.format(x2.shape))
        # print('x3.shape:{}'.format(x3.shape))
        # print('x4.shape:{}'.format(x4.shape))

        # temp1 = []
        # temp1.append(x1)
        # temp1.append(x2)
        # temp1.append(x3)
        # temp1.append(x4)

        t0 = time.time()
        x1 = self.embed_1(x1)
        # shortcut = x1
        x1, attn_sc_1 = self.mhsa_1(x1)
        x1_ = x1 + self.drop_path_1(x1)

        x2 = self.embed_2(x2)
        # shortcut = x2
        x2, attn_sc_2 = self.mhsa_2(x2)
        x2_ = x2 + self.drop_path_2(x2)

        x3 = self.embed_3(x3)
        # shortcut = x3
        x3, attn_sc_3 = self.mhsa_3(x3)
        x3_ = x3 + self.drop_path_3(x3)

        x4 = self.embed_4(x4)
        # shortcut = x4
        x4, attn_sc_4 = self.mhsa_4(x4)
        x4_ = x4 + self.drop_path_4(x4)
        # print('attn time:{:.6f}'.format(time.time() - t0))

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # print('x4_.shape:{}'.format(x4_.shape))

        t0 = time.time()
        x1_ = self.unfold_1(x1_)
        x2_ = self.unfold_2(x2_)
        x3_ = self.unfold_3(x3_)
        x4_ = self.unfold_4(x4_)
        # print('unfold time:{:.6f}'.format(time.time() - t0))

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # print('x4_.shape:{}'.format(x4_.shape))

        x1_ = F.interpolate(x1_, scale_factor=8.0, mode='bilinear')
        x2_ = F.interpolate(x2_, scale_factor=4.0, mode='bilinear')
        x3_ = F.interpolate(x3_, scale_factor=2.0, mode='bilinear')

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # print('x4_.shape:{}'.format(x4_.shape))

        # temp2 = []
        # temp2.append(x1_)
        # temp2.append(x2_)
        # temp2.append(x3_)
        # temp2.append(x4_)

        features.append(x1_)
        features.append(x2_)
        features.append(x3_)
        features.append(x4_)

        return features  # , temp1, temp2


class EdgeNetV1M(nn.Module):
    def __init__(self, reduction=1, img_shape=(480, 640)):
        super(EdgeNetV1M, self).__init__()
        self.reduction = reduction
        h, w = img_shape[0], img_shape[1]

        input_feats = 3
        output_feats = 32 // reduction
        # print('input_feats:{} output_feats:{}'.format(input_feats, output_feats))
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/4, w/4]

        self.drop_path_1 = TFDropPath(0.025)
        ps = 8
        h_ = h // 4
        w_ = w // 4
        conv_output_channels = output_feats
        self.embed_1 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_1 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_1 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_1 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        input_feats = output_feats
        output_feats = 64 // reduction
        # print('input_feats:{} output_feats:{}'.format(input_feats, output_feats))
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]

        self.drop_path_2 = TFDropPath(0.050)
        ps = 4
        h_ = h // 8
        w_ = w // 8
        conv_output_channels = output_feats
        self.embed_2 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_2 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_2 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_2 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        input_feats = output_feats
        output_feats = 128 // reduction
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/16, w/16]

        self.drop_path_3 = TFDropPath(0.075)
        ps = 2
        h_ = h // 16
        w_ = w // 16
        conv_output_channels = output_feats
        self.embed_3 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_3 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_3 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_3 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        input_feats = output_feats
        output_feats = 256 // reduction
        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 256, h/32, w/32]

        self.drop_path_4 = TFDropPath(0.100)
        ps = 1
        h_ = h // 32
        w_ = w // 32
        conv_output_channels = output_feats
        self.embed_4 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_4 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_4 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_4 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

    def forward(self, x):
        features = []
        x1 = self.edge_encoder_1(x)  # [B, 32, 120, 160]
        x2 = self.edge_encoder_2(x1)  # [B, 64, 60, 80]
        x3 = self.edge_encoder_3(x2)  # [B, 128, 30, 40]
        x4 = self.edge_encoder_4(x3)  # [B, 256, 15, 20]

        # t0 = time.time()
        x1 = self.embed_1(x1)
        shortcut = x1
        x1, attn_sc_1 = self.mhsa_1(x1)
        x1_ = shortcut + self.drop_path_1(x1)

        x2 = self.embed_2(x2)
        shortcut = x2
        x2, attn_sc_2 = self.mhsa_2(x2)
        x2_ = shortcut + self.drop_path_2(x2)

        x3 = self.embed_3(x3)
        shortcut = x3
        x3, attn_sc_3 = self.mhsa_3(x3)
        x3_ = shortcut + self.drop_path_3(x3)

        x4 = self.embed_4(x4)
        shortcut = x4
        x4, attn_sc_4 = self.mhsa_4(x4)
        x4_ = shortcut + self.drop_path_4(x4)
        # print('attn time:{:.6f}'.format(time.time() - t0))

        # t0 = time.time()
        x1_ = self.unfold_1(x1_)
        x2_ = self.unfold_2(x2_)
        x3_ = self.unfold_3(x3_)
        x4_ = self.unfold_4(x4_)
        # print('unfold time:{:.6f}'.format(time.time() - t0))

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # print('x4_.shape:{}'.format(x4_.shape))

        features.append(x1_)
        features.append(x2_)
        features.append(x3_)
        features.append(x4_)

        return features  # , temp1, temp2


class EdgeNetV2(nn.Module):
    def __init__(self, reduction=1, img_shape=(480, 640)):
        super(EdgeNetV2, self).__init__()
        self.reduction = reduction
        h, w = img_shape[0], img_shape[1]

        input_feats = 3
        output_feats = 32 // reduction
        # print('input_feats:{} output_feats:{}'.format(input_feats, output_feats))
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/4, w/4]

        self.drop_path_1 = TFDropPath(0.025)
        ps = 8
        h_ = h // 4
        w_ = w // 4
        conv_output_channels = output_feats
        self.embed_1 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.qk_1 = SimilarityQK(dim=ps * ps * conv_output_channels)
        # self.unfold_1 = PatchEmbedUnfold(feature_map_size=(h//4, w//4), patch_size=ps)

        input_feats = output_feats
        output_feats = 64 // reduction
        # print('input_feats:{} output_feats:{}'.format(input_feats, output_feats))
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]

        self.drop_path_2 = TFDropPath(0.050)
        ps = 4
        h_ = h // 8
        w_ = w // 8
        conv_output_channels = output_feats
        self.embed_2 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.qk_2 = SimilarityQK(dim=ps * ps * conv_output_channels)
        # self.unfold_2 = PatchEmbedUnfold(feature_map_size=(h//8, w//8), patch_size=ps)

        input_feats = output_feats
        output_feats = 128 // reduction
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/16, w/16]

        self.drop_path_3 = TFDropPath(0.075)
        ps = 2
        h_ = h // 16
        w_ = w // 16
        conv_output_channels = output_feats
        self.embed_3 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.qk_3 = SimilarityQK(dim=ps * ps * conv_output_channels)
        # self.unfold_3 = PatchEmbedUnfold(feature_map_size=(h//16, w//16), patch_size=ps)

        input_feats = output_feats
        output_feats = 256 // reduction
        self.edge_encoder_4 = nn.Sequential(
            nn.Conv2d(input_feats, output_feats // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feats // 2),
            nn.Conv2d(output_feats // 2, output_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_feats),
            nn.ReLU(inplace=True),
        )  # [b, 256, h/32, w/32]

        self.drop_path_4 = TFDropPath(0.100)
        ps = 1
        h_ = h // 32
        w_ = w // 32
        conv_output_channels = output_feats
        self.embed_4 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.qk_4 = SimilarityQK(dim=ps * ps * conv_output_channels)
        # self.unfold_4 = PatchEmbedUnfold(feature_map_size=(h//32, w//32), patch_size=ps)

    def forward(self, x):
        features_list = []
        similarity_qk_list = []

        x1 = self.edge_encoder_1(x)  # [B, 32, 120, 160]
        x2 = self.edge_encoder_2(x1)  # [B, 64, 60, 80]
        x3 = self.edge_encoder_3(x2)  # [B, 128, 30, 40]
        x4 = self.edge_encoder_4(x3)  # [B, 256, 15, 20]

        features_list.append(x1)
        features_list.append(x2)
        features_list.append(x3)
        features_list.append(x4)

        # print('======== class name: EdgeNetV2 ========')
        # print('x1.shape:{}'.format(x1.shape))
        # print('x2.shape:{}'.format(x2.shape))
        # print('x3.shape:{}'.format(x3.shape))
        # print('x4.shape:{}'.format(x4.shape))

        # t0 = time.time()
        x1_ = self.embed_1(x1)
        x1_ = self.qk_1(x1_)

        x2_ = self.embed_2(x2)
        x2_ = self.qk_2(x2_)

        x3_ = self.embed_3(x3)
        x3_ = self.qk_3(x3_)

        x4_ = self.embed_4(x4)
        x4_ = self.qk_4(x4_)
        # print('attn time:{:.6f}'.format(time.time() - t0))

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # print('x4_.shape:{}'.format(x4_.shape))

        similarity_qk_list.append(x1_)
        similarity_qk_list.append(x2_)
        similarity_qk_list.append(x3_)
        similarity_qk_list.append(x4_)

        return features_list, similarity_qk_list


class EdgeNetV3(nn.Module):
    """
    参考PPT P39
    """

    def __init__(self, reduction=1, img_shape=(480, 640)):
        super(EdgeNetV3, self).__init__()

        h, w = img_shape[0], img_shape[1]

        # NOTE: Params(0.25M)
        intermediate_feats = 16 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, transition_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]

        self.drop_path_1 = TFDropPath(0.0333)
        ps = 8
        h_ = h // 2
        w_ = w // 2
        conv_output_channels = intermediate_feats
        self.embed_1 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.mhsa_1 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_1 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        conv_output_channels = intermediate_feats
        intermediate_feats = 32 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/4, w/4]

        self.drop_path_2 = TFDropPath(0.0666)
        ps = 4
        h_ = h // 4
        w_ = w // 4
        conv_output_channels = intermediate_feats
        # print('conv_output_channels:{}'.format(conv_output_channels))
        self.embed_2 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.mhsa_2 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_2 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/8, w/8]

        self.drop_path_3 = TFDropPath(0.1000)
        ps = 2
        h_ = h // 8
        w_ = w // 8
        conv_output_channels = intermediate_feats
        # print('conv_output_channels:{}'.format(conv_output_channels))
        self.embed_3 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.mhsa_3 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_3 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

    def forward(self, x):
        features = []
        x1 = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        x2 = self.edge_encoder_2(x1)  # [B, 64, 120, 160]
        x3 = self.edge_encoder_3(x2)  # [B, 64, 60, 80]

        # t0 = time.time()
        x1 = self.embed_1(x1)
        shortcut = x1
        x1, attn_sc_1 = self.mhsa_1(x1)
        x1_ = shortcut + self.drop_path_1(x1)

        x2 = self.embed_2(x2)
        shortcut = x2
        x2, attn_sc_2 = self.mhsa_2(x2)
        x2_ = shortcut + self.drop_path_2(x2)

        x3 = self.embed_3(x3)
        shortcut = x3
        x3, attn_sc_3 = self.mhsa_3(x3)
        x3_ = shortcut + self.drop_path_3(x3)
        # print('attn time:{:.6f}'.format(time.time() - t0))

        # t0 = time.time()
        x1_ = self.unfold_1(x1_)
        x2_ = self.unfold_2(x2_)
        x3_ = self.unfold_3(x3_)
        # print('unfold time:{:.6f}'.format(time.time() - t0))

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # exit()

        x1_ = F.interpolate(x1_, scale_factor=8.0, mode='bilinear')
        x2_ = F.interpolate(x2_, scale_factor=4.0, mode='bilinear')
        x3_ = F.interpolate(x3_, scale_factor=2.0, mode='bilinear')

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # exit()

        features.append(x1_)
        features.append(x2_)
        features.append(x3_)

        return features


class EdgeNetV4(nn.Module):
    """
    参考PPT P48
    """

    def __init__(self, reduction=1, img_shape=(480, 640)):
        super(EdgeNetV4, self).__init__()

        h, w = img_shape[0], img_shape[1]

        # NOTE: Params(0.25M)
        intermediate_feats = 32 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, transition_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/4, w/4]

        conv_output_channels = intermediate_feats
        intermediate_feats = 128 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/8, w/8]

    def forward(self, x):
        features = []

        x1 = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        x2 = self.edge_encoder_2(x1)  # [B, 64, 120, 160]
        x3 = self.edge_encoder_3(x2)  # [B, 64, 60, 80]

        features.append(x1)
        features.append(x2)
        features.append(x3)

        return features


class EdgeNetV5(nn.Module):
    """
    参考PPT P48
    """

    def __init__(self, reduction=1, img_shape=(480, 640)):
        super(EdgeNetV5, self).__init__()

        h, w = img_shape[0], img_shape[1]

        # NOTE: Params(0.25M)
        intermediate_feats = 32 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, transition_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/4, w/4]

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 128, h/8, w/8]

    def forward(self, x):
        features = []

        x1 = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        x2 = self.edge_encoder_2(x1)  # [B, 64, 120, 160]
        x3 = self.edge_encoder_3(x2)  # [B, 64, 60, 80]

        features.append(x1)
        features.append(x2)
        features.append(x3)

        return features


class EdgeNetV5Attn(nn.Module):
    """
    参考PPT P52
    """

    def __init__(self, reduction=1, img_shape=(480, 640)):
        super(EdgeNetV5Attn, self).__init__()

        h, w = img_shape[0], img_shape[1]

        # NOTE: Params(0.25M)
        intermediate_feats = 32 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, transition_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/2, w/2]

        self.avg_pool1 = nn.AvgPool2d(kernel_size=5, stride=4, padding=1)

        num_heads = 2
        self.attn1 = AttnBlock(input_dim=intermediate_feats,
                               num_heads=num_heads,
                               qkv_dim_per_head=intermediate_feats // num_heads,
                               drop_path_ratio=0.0333)

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/4, w/4]

        self.avg_pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        num_heads = 4
        self.attn2 = AttnBlock(input_dim=intermediate_feats,
                               num_heads=num_heads,
                               qkv_dim_per_head=intermediate_feats // num_heads,
                               drop_path_ratio=0.0666)

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]

        num_heads = 4
        self.attn3 = AttnBlock(input_dim=intermediate_feats,
                               num_heads=num_heads,
                               qkv_dim_per_head=intermediate_feats // num_heads,
                               drop_path_ratio=0.0999)

    def forward(self, x):
        features = []

        x1 = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        x2 = self.edge_encoder_2(x1)  # [B, 64, 120, 160]
        x3 = self.edge_encoder_3(x2)  # [B, 64, 60, 80]

        x1 = self.avg_pool1(x1)
        x2 = self.avg_pool2(x2)

        # print('x1.shape:{}'.format(x1.shape))
        # print('x2.shape:{}'.format(x2.shape))

        x1 = self.attn1(x1)
        x2 = self.attn2(x2)
        x3 = self.attn3(x3)

        x1 = F.interpolate(x1, scale_factor=4.0, mode='bilinear')
        x2 = F.interpolate(x2, scale_factor=2.0, mode='bilinear')

        # print('x1.shape:{}'.format(x1.shape))
        # print('x2.shape:{}'.format(x2.shape))
        # print('x3.shape:{}'.format(x3.shape))

        features.append(x1)
        features.append(x2)
        features.append(x3)

        return features


class EdgeNetV6(nn.Module):
    """
    参考PPT P48
    """

    def __init__(self, reduction=1):
        super(EdgeNetV6, self).__init__()

        # NOTE: Params(0.25M)
        intermediate_feats = 32 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_1 = nn.Sequential(
            nn.Conv2d(3, transition_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 32, h/4, w/4]

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_2 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [32, 16, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [64, 32, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/8, w/8]

        conv_output_channels = intermediate_feats
        intermediate_feats = 64 // reduction
        transition_feats = intermediate_feats // 2
        self.edge_encoder_3 = nn.Sequential(
            nn.Conv2d(conv_output_channels, transition_feats, kernel_size=3, stride=1, padding=1),
            # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(transition_feats),
            nn.Conv2d(transition_feats, intermediate_feats, kernel_size=3, stride=2, padding=1),
            # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(intermediate_feats),
            nn.ReLU(inplace=True),
        )  # [b, 64, h/16, w/16]

    def forward(self, x):
        features = []

        x1 = self.edge_encoder_1(x)  # [B, 16, 240, 320]
        x2 = self.edge_encoder_2(x1)  # [B, 64, 120, 160]
        x3 = self.edge_encoder_3(x2)  # [B, 64, 60, 80]

        features.append(x1)
        features.append(x2)
        features.append(x3)

        return features


class BackboneMultiScaleAttnExtractor1(nn.Module):
    """
    Backbone Multi Scale Attention Extractor (BMSAE1)
    """

    def __init__(self,
                 img_shape=(480, 640),
                 depths=4,
                 feats_list=[32, 64, 128, 256],
                 ps_list=[8, 4, 2, 1]):
        super(BackboneMultiScaleAttnExtractor1, self).__init__()
        h, w = img_shape[0], img_shape[1]

        # self.drop_path = nn.ModuleList()
        # self.embed = nn.ModuleList()
        # self.mhsa = nn.ModuleList()
        # self.unfold = nn.ModuleList()

        self.drop_path_1 = TFDropPath(0.025)
        ps = 8
        h_ = h // 4
        w_ = w // 4
        conv_output_channels = feats_list[0]
        self.embed_1 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_1 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_1 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_1 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        self.drop_path_2 = TFDropPath(0.050)
        ps = 4
        h_ = h // 8
        w_ = w // 8
        conv_output_channels = feats_list[1]
        self.embed_2 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_2 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_2 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_2 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        self.drop_path_3 = TFDropPath(0.075)
        ps = 2
        h_ = h // 16
        w_ = w // 16
        conv_output_channels = feats_list[2]
        self.embed_3 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_3 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_3 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_3 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        self.drop_path_4 = TFDropPath(0.100)
        ps = 1
        h_ = h // 32
        w_ = w // 32
        conv_output_channels = feats_list[3]
        self.embed_4 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        # self.mhsa_4 = Attention(dim=ps * ps * conv_output_channels, visualization=False)
        self.mhsa_4 = MyAttention(dim=ps * ps * conv_output_channels, visualization=True)
        self.unfold_4 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

    def forward(self, features):
        features_ = []
        # 最后一层要调换到第二层位置
        x1, x2, x3, x4 = features[0], features[3], features[1], features[2]

        # print('x1.shape:{}'.format(x1.shape))
        # print('x2.shape:{}'.format(x2.shape))
        # print('x3.shape:{}'.format(x3.shape))
        # print('x4.shape:{}'.format(x4.shape))

        # t0 = time.time()
        x1 = self.embed_1(x1)
        x1, attn_sc_1 = self.mhsa_1(x1)
        x1_ = x1 + self.drop_path_1(x1)

        x2 = self.embed_2(x2)
        x2, attn_sc_2 = self.mhsa_2(x2)
        x2_ = x2 + self.drop_path_2(x2)

        x3 = self.embed_3(x3)
        x3, attn_sc_3 = self.mhsa_3(x3)
        x3_ = x3 + self.drop_path_3(x3)

        x4 = self.embed_4(x4)
        x4, attn_sc_4 = self.mhsa_4(x4)
        x4_ = x4 + self.drop_path_4(x4)
        # print('attn time:{:.6f}'.format(time.time() - t0))

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # print('x4_.shape:{}'.format(x4_.shape))

        # t0 = time.time()
        x1_ = self.unfold_1(x1_)
        x2_ = self.unfold_2(x2_)
        x3_ = self.unfold_3(x3_)
        x4_ = self.unfold_4(x4_)
        # print('unfold time:{:.6f}'.format(time.time() - t0))

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # print('x4_.shape:{}'.format(x4_.shape))

        x1_ = F.interpolate(x1_, scale_factor=8.0, mode='bilinear')
        x2_ = F.interpolate(x2_, scale_factor=4.0, mode='bilinear')
        x3_ = F.interpolate(x3_, scale_factor=2.0, mode='bilinear')

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # print('x4_.shape:{}'.format(x4_.shape))

        features_.append(x1_)
        features_.append(x2_)
        features_.append(x3_)
        features_.append(x4_)

        return features_


# class BBMSAE2(nn.Module):
#     """
#     Backbone Multi Sacle Attention Extractor
#     """
#     def __init__(self,
#                  img_shape=(480, 640),
#                  feats_list=[32, 64, 128, 256],
#                  ps_list=[8, 4, 2, 1]):
#         super(BBMSAE2, self).__init__()
#         h, w = img_shape[0], img_shape[1]
#
#         # 8的整数倍 最后一层特征图只有15*20的尺寸 embedding_nums为1*1*8=8 而多头注意力的head数设置为了8
#         essence_feature_nums = 8
#         self.squeeze_conv1 = nn.Conv2d(feats_list[0], essence_feature_nums, kernel_size=1, stride=1, padding=0)
#         self.squeeze_conv2 = nn.Conv2d(feats_list[1], essence_feature_nums, kernel_size=1, stride=1, padding=0)
#         self.squeeze_conv3 = nn.Conv2d(feats_list[2], essence_feature_nums, kernel_size=1, stride=1, padding=0)
#         self.squeeze_conv4 = nn.Conv2d(feats_list[3], essence_feature_nums, kernel_size=1, stride=1, padding=0)
#
#         self.drop_path_1 = TFDropPath(0.025)
#         ps = 8
#         h_ = h//4
#         w_ = w//4
#         conv_output_channels = essence_feature_nums
#         self.embed_1 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
#         self.mhsa_1 = MyAttention(dim=ps*ps*conv_output_channels, visualization=True)
#         self.unfold_1 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)
#
#         self.drop_path_2 = TFDropPath(0.050)
#         ps = 4
#         h_ = h//8
#         w_ = w//8
#         conv_output_channels = essence_feature_nums
#         self.embed_2 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
#         self.mhsa_2 = MyAttention(dim=ps*ps*conv_output_channels, visualization=True)
#         self.unfold_2 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)
#
#         self.drop_path_3 = TFDropPath(0.075)
#         ps = 2
#         h_ = h//16
#         w_ = w//16
#         conv_output_channels = essence_feature_nums
#         self.embed_3 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
#         self.mhsa_3 = MyAttention(dim=ps*ps*conv_output_channels, visualization=True)
#         self.unfold_3 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)
#
#         self.drop_path_4 = TFDropPath(0.100)
#         ps = 1
#         h_ = h//32
#         w_ = w//32
#         conv_output_channels = essence_feature_nums
#         self.embed_4 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
#         self.mhsa_4 = MyAttention(dim=ps*ps*conv_output_channels, visualization=True)
#         self.unfold_4 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)
#
#     def forward(self, features):
#         features_ = []
#         similarity_qk_list = []
#         # 最后一层要调换到第二层位置
#         x1, x2, x3, x4 = features[0], features[3], features[1], features[2]
#
#         print('x1.shape:{}'.format(x1.shape))
#         print('x2.shape:{}'.format(x2.shape))
#         print('x3.shape:{}'.format(x3.shape))
#         print('x4.shape:{}'.format(x4.shape))
#
#         x1 = self.squeeze_conv1(x1)
#         x2 = self.squeeze_conv2(x2)
#         x3 = self.squeeze_conv3(x3)
#         x4 = self.squeeze_conv4(x4)
#
#         # t0 = time.time()
#         x1 = self.embed_1(x1)
#         x1, attn_sc_1, similarity_qk1 = self.mhsa_1(x1)
#         x1_ = x1 + self.drop_path_1(x1)
#
#         x2 = self.embed_2(x2)
#         x2, attn_sc_2, similarity_qk2 = self.mhsa_2(x2)
#         x2_ = x2 + self.drop_path_2(x2)
#
#         x3 = self.embed_3(x3)
#         x3, attn_sc_3, similarity_qk3 = self.mhsa_3(x3)
#         x3_ = x3 + self.drop_path_3(x3)
#
#         x4 = self.embed_4(x4)
#         x4, attn_sc_4, similarity_qk4 = self.mhsa_4(x4)
#         x4_ = x4 + self.drop_path_4(x4)
#         # print('attn time:{:.6f}'.format(time.time() - t0))
#
#         # [B, num_heads, num_patches, num_patches] = [b, 8, 15*20, 15*20] = [b, 8, 300, 300]
#         similarity_qk_list.append(similarity_qk1)
#         similarity_qk_list.append(similarity_qk2)
#         similarity_qk_list.append(similarity_qk3)
#         similarity_qk_list.append(similarity_qk4)
#
#         print('similarity_qk1.shape:{}'.format(similarity_qk1.shape))
#         print('similarity_qk2.shape:{}'.format(similarity_qk2.shape))
#         print('similarity_qk3.shape:{}'.format(similarity_qk3.shape))
#         print('similarity_qk4.shape:{}'.format(similarity_qk4.shape))
#
#         print('x1_.shape:{}'.format(x1_.shape))
#         print('x2_.shape:{}'.format(x2_.shape))
#         print('x3_.shape:{}'.format(x3_.shape))
#         print('x4_.shape:{}'.format(x4_.shape))
#
#         # t0 = time.time()
#         x1_ = self.unfold_1(x1_)
#         x2_ = self.unfold_2(x2_)
#         x3_ = self.unfold_3(x3_)
#         x4_ = self.unfold_4(x4_)
#         # print('unfold time:{:.6f}'.format(time.time() - t0))
#
#         print('x1_.shape:{}'.format(x1_.shape))
#         print('x2_.shape:{}'.format(x2_.shape))
#         print('x3_.shape:{}'.format(x3_.shape))
#         print('x4_.shape:{}'.format(x4_.shape))
#
#         exit()
#
#         x1_ = F.interpolate(x1_, scale_factor=8.0, mode='bilinear')
#         x2_ = F.interpolate(x2_, scale_factor=4.0, mode='bilinear')
#         x3_ = F.interpolate(x3_, scale_factor=2.0, mode='bilinear')
#
#         # print('x1_.shape:{}'.format(x1_.shape))
#         # print('x2_.shape:{}'.format(x2_.shape))
#         # print('x3_.shape:{}'.format(x3_.shape))
#         # print('x4_.shape:{}'.format(x4_.shape))
#
#         features_.append(x1_)
#         features_.append(x2_)
#         features_.append(x3_)
#         features_.append(x4_)
#
#         return features_, similarity_qk_list


class BackboneMultiScaleAttnExtractor2(nn.Module):
    """
    Backbone Multi Scale Attention Extractor (BMSAE2)
    """

    def __init__(self,
                 img_shape=(480, 640),
                 feats_list=[32, 64, 128, 256],
                 ps_list=[8, 4, 2, 1]):
        super(BackboneMultiScaleAttnExtractor2, self).__init__()
        h, w = img_shape[0], img_shape[1]

        # 8的整数倍 最后一层特征图只有15*20的尺寸 embedding_nums为1*1*8=8 而多头注意力的head数设置为了8
        essence_feature_nums = 8
        self.squeeze_conv1 = nn.Conv2d(feats_list[0], essence_feature_nums, kernel_size=1, stride=1, padding=0)
        self.squeeze_conv2 = nn.Conv2d(feats_list[1], essence_feature_nums, kernel_size=1, stride=1, padding=0)
        self.squeeze_conv3 = nn.Conv2d(feats_list[2], essence_feature_nums, kernel_size=1, stride=1, padding=0)
        self.squeeze_conv4 = nn.Conv2d(feats_list[3], essence_feature_nums, kernel_size=1, stride=1, padding=0)

        self.drop_path_1 = TFDropPath(0.025)
        ps = 8
        h_ = h // 4
        w_ = w // 4
        conv_output_channels = essence_feature_nums
        self.embed_1 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.qk_1 = SimilarityQK(dim=ps * ps * conv_output_channels)
        # self.unfold_1 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        self.drop_path_2 = TFDropPath(0.050)
        ps = 4
        h_ = h // 8
        w_ = w // 8
        conv_output_channels = essence_feature_nums
        self.embed_2 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.qk_2 = SimilarityQK(dim=ps * ps * conv_output_channels)
        # self.unfold_2 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        self.drop_path_3 = TFDropPath(0.075)
        ps = 2
        h_ = h // 16
        w_ = w // 16
        conv_output_channels = essence_feature_nums
        self.embed_3 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.qk_3 = SimilarityQK(dim=ps * ps * conv_output_channels)
        # self.unfold_3 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

        self.drop_path_4 = TFDropPath(0.100)
        ps = 1
        h_ = h // 32
        w_ = w // 32
        conv_output_channels = essence_feature_nums
        self.embed_4 = TFPatchEmbed(img_size=(h_, w_), patch_size=ps, in_c=conv_output_channels)
        self.qk_4 = SimilarityQK(dim=ps * ps * conv_output_channels)
        # self.unfold_4 = PatchEmbedUnfold(feature_map_size=(h_, w_), patch_size=ps)

    def forward(self, features):
        similarity_qk_list = []
        # 最后一层要调换到第二层位置
        x1, x2, x3, x4 = features[0], features[3], features[1], features[2]

        # print('======== class name: BBMSAE2 ========')
        # print('x1.shape:{}'.format(x1.shape))
        # print('x2.shape:{}'.format(x2.shape))
        # print('x3.shape:{}'.format(x3.shape))
        # print('x4.shape:{}'.format(x4.shape))

        x1 = self.squeeze_conv1(x1)
        x2 = self.squeeze_conv2(x2)
        x3 = self.squeeze_conv3(x3)
        x4 = self.squeeze_conv4(x4)

        # t0 = time.time()
        x1 = self.embed_1(x1)
        x1_ = self.qk_1(x1)

        x2 = self.embed_2(x2)
        x2_ = self.qk_2(x2)

        x3 = self.embed_3(x3)
        x3_ = self.qk_3(x3)

        x4 = self.embed_4(x4)
        x4_ = self.qk_4(x4)
        # print('attn time:{:.6f}'.format(time.time() - t0))

        # [B, num_heads, num_patches, num_patches] = [b, 8, 15*20, 15*20] = [b, 8, 300, 300]
        similarity_qk_list.append(x1_)
        similarity_qk_list.append(x2_)
        similarity_qk_list.append(x3_)
        similarity_qk_list.append(x4_)

        # print('x1_.shape:{}'.format(x1_.shape))
        # print('x2_.shape:{}'.format(x2_.shape))
        # print('x3_.shape:{}'.format(x3_.shape))
        # print('x4_.shape:{}'.format(x4_.shape))

        return similarity_qk_list


class BackboneMultiScaleAttnExtractor3(nn.Module):
    """
    Backbone Multi Scale Attention Extractor (BMSAE3)
    """

    def __init__(self,
                 img_shape=(480, 640),
                 feats_list=[32, 64, 64]):
        super(BackboneMultiScaleAttnExtractor3, self).__init__()
        h, w = img_shape[0], img_shape[1]
        self.feats_list = feats_list

        self.ppa = PyramidPoolAgg()

        depths = 4
        drop_path_rate = 0.1
        embed_dim = sum(feats_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        norm_cfg = dict(type='BN2d', requires_grad=True)
        act_layer = nn.ReLU6
        self.trans = BasicLayer(
            block_num=depths,  # 4
            embedding_dim=embed_dim,  # 160
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
                SemanticInjectionModule(feats_list[i], feats_list[i], norm_cfg=norm_cfg, activations=None)
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


class DecoderBlockV1(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 attn_type=None,
                 scale_factor=4.0,
                 is_lastblock=False):
        super(DecoderBlockV1, self).__init__()
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
                nn.Conv2d(edge_feats, edge_feats // 2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(edge_feats // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_feats // 2, 1, kernel_size=1, stride=1, padding=0),
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


class DecoderV1(nn.Module):
    def __init__(self, attn_type=None):
        super(DecoderV1, self).__init__()

        # print('Decoder attn_type:{}'.format(attn_type))

        self.decoder_block_1 = DecoderBlockV1(1, 256, 256, output_feats=128, attn_type=attn_type,
                                              scale_factor=2.0)
        self.decoder_block_2 = DecoderBlockV1(2, 128, 128, output_feats=64, target_feats=128, attn_type=attn_type,
                                              scale_factor=2.0)
        self.decoder_block_3 = DecoderBlockV1(3, 64, 64, output_feats=32, target_feats=64, attn_type=attn_type,
                                              scale_factor=2.0)
        self.decoder_block_4 = DecoderBlockV1(4, 32, 32, output_feats=1, target_feats=32, attn_type=attn_type,
                                              scale_factor=4.0, is_lastblock=True)

    def forward(self, features, edges, x=None):
        # for i in range(len(features)):
        #     print('features[{}].shape:{}'.format(i, features[i].shape))
        # for i in range(len(edges)):
        #     print('edges[{}].shape:{}'.format(i, edges[i].shape))

        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)
        y = self.decoder_block_4(features[-4], edges[-4], target=y)

        return y


class AttnFuseBlock(nn.Module):
    def __init__(self, conv_output_channels, ps, attn_drop_ratio=0., proj_drop_ratio=0., tf_drop_ratio=0.,
                 img_shape=(120, 160)):
        """

        Args:
            conv_output_channels:
            ps: 这一层decoder block对应的该层特征提取网络所采用的ps
            attn_drop_ratio:
            img_shape: 这一层decoder block对应的该层特征提取网络输出的尺寸
        """
        super(AttnFuseBlock, self).__init__()
        h, w = img_shape[0], img_shape[1]
        self.embedding_dims = ps * ps * conv_output_channels

        self.embed = TFPatchEmbed(img_size=(h, w), patch_size=ps, in_c=conv_output_channels)
        self.get_v = GetV(dim=self.embedding_dims)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(self.embedding_dims, self.embedding_dims)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.drop_path = TFDropPath(tf_drop_ratio)  # 逐级递增
        self.embed_unfold = PatchEmbedUnfold(feature_map_size=img_shape, patch_size=ps, in_c=conv_output_channels)

    def forward(self, feature, qk1, qk2):
        B, N = feature.shape[0], qk1.shape[2]
        feature_patch_embedding = self.embed(feature)
        # print('feature_patch_embedding.shape:{}'.format(feature_patch_embedding.shape))
        feature_v = self.get_v(feature_patch_embedding)

        qk_fused = qk1 + qk2
        qk_fused = qk_fused.softmax(dim=-1)
        qk_fused = self.attn_drop(qk_fused)

        # @: multiply -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches, total_embed_dim]
        x = (qk_fused @ feature_v).transpose(1, 2).reshape(B, N, self.embedding_dims)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.drop_path(x)  # 仿照tf_block中的MHSA之后的dropout 但是没有残差连接

        x = self.embed_unfold(x)

        return x


class DecoderBlockV2(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 scale_factor=4.0,
                 ps=1,
                 tf_drop_ratio=0.,
                 img_shape=(15, 20),
                 is_lastblock=False):
        super(DecoderBlockV2, self).__init__()
        self.block_sn = block_sn
        self.is_lastblock = is_lastblock
        self.scale_factor = scale_factor

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        intermediate_feats = n_feats // 4
        self.squeeze_conv = nn.Conv2d(n_feats, intermediate_feats, kernel_size=1, stride=1, padding=0)

        self.attn_fuse = AttnFuseBlock(conv_output_channels=intermediate_feats, ps=ps,
                                       tf_drop_ratio=tf_drop_ratio, img_shape=img_shape)

        self.conv1 = nn.Sequential(
            nn.Conv2d(intermediate_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        if is_lastblock is False:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, edge_feats // 2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(edge_feats // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_feats // 2, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, rgb_feature, edge_feature, qk1, qk2, target=None):
        if target is None:
            x = torch.cat((rgb_feature, edge_feature), dim=1)
        else:
            x = torch.cat((rgb_feature, edge_feature, target), dim=1)

        x = self.squeeze_conv(x)
        x = self.attn_fuse(feature=x, qk1=qk1, qk2=qk2)
        x = self.conv1(x)
        # print('x.shape:{}'.format(x.shape))
        x = x + edge_feature
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.conv2(x)

        return x


class DecoderV2(nn.Module):
    def __init__(self):
        super(DecoderV2, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, 0.1, 4)]
        self.decoder_block1 = DecoderBlockV2(block_sn=1, rgb_feats=256, edge_feats=256, output_feats=128,
                                             scale_factor=2.0, ps=1, tf_drop_ratio=dpr[0],
                                             img_shape=(15, 20))
        self.decoder_block2 = DecoderBlockV2(block_sn=2, rgb_feats=128, edge_feats=128, output_feats=64,
                                             target_feats=128, scale_factor=2.0, ps=2, tf_drop_ratio=dpr[1],
                                             img_shape=(30, 40))
        self.decoder_block3 = DecoderBlockV2(block_sn=3, rgb_feats=64, edge_feats=64, output_feats=32,
                                             target_feats=64, scale_factor=2.0, ps=4, tf_drop_ratio=dpr[2],
                                             img_shape=(60, 80))
        self.decoder_block4 = DecoderBlockV2(block_sn=4, rgb_feats=32, edge_feats=32, output_feats=1,
                                             target_feats=32, scale_factor=4.0, ps=8, tf_drop_ratio=dpr[3],
                                             img_shape=(120, 160), is_lastblock=True)

    def forward(self, rgb_features, edge_features, rgb_qk_list, edge_qk_list):
        x1, x2, x3, x4 = rgb_features[0], rgb_features[3], rgb_features[1], rgb_features[2]

        x_list = []
        x_list.append(rgb_features[0])
        x_list.append(rgb_features[3])
        x_list.append(rgb_features[1])
        x_list.append(rgb_features[2])

        # for i in range(len(rgb_features)):
        #     print('rgb_features[{}].shape:{}'.format(i, rgb_features[i].shape))
        # for i in range(len(x_list)):
        #     print('x_list[{}].shape:{}'.format(i, x_list[i].shape))
        # for i in range(len(edge_features)):
        #     print('edge_features[{}].shape:{}'.format(i, edge_features[i].shape))
        # for i in range(len(rgb_qk_list)):
        #     print('rgb_qk_list[{}].shape:{}'.format(i, rgb_qk_list[i].shape))
        # for i in range(len(edge_qk_list)):
        #     print('edge_qk_list[{}].shape:{}'.format(i, edge_qk_list[i].shape))

        # y = self.decoder_block1(x4, edge_features[-1], rgb_qk_list[-1], edge_qk_list[-1])
        # y = self.decoder_block2(x3, edge_features[-2], rgb_qk_list[-2], edge_qk_list[-2], target=y)
        # y = self.decoder_block3(x2, edge_features[-3], rgb_qk_list[-3], edge_qk_list[-3], target=y)
        # y = self.decoder_block4(x1, edge_features[-4], rgb_qk_list[-4], edge_qk_list[-4], target=y)

        y = self.decoder_block1(x_list[-1], edge_features[-1], rgb_qk_list[-1], edge_qk_list[-1])
        y = self.decoder_block2(x_list[-2], edge_features[-2], rgb_qk_list[-2], edge_qk_list[-2], target=y)
        y = self.decoder_block3(x_list[-3], edge_features[-3], rgb_qk_list[-3], edge_qk_list[-3], target=y)
        y = self.decoder_block4(x_list[-4], edge_features[-4], rgb_qk_list[-4], edge_qk_list[-4], target=y)

        return y


class DecoderBlockV2M1(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 scale_factor=4.0,
                 ps=1,
                 tf_drop_ratio=0.,
                 img_shape=(15, 20),
                 is_lastblock=False):
        super(DecoderBlockV2M1, self).__init__()
        self.block_sn = block_sn
        self.is_lastblock = is_lastblock
        self.scale_factor = scale_factor

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        intermediate_feats = n_feats // 4
        self.squeeze_conv = nn.Conv2d(n_feats, intermediate_feats, kernel_size=1, stride=1, padding=0)

        self.attn_fuse = AttnFuseBlock(conv_output_channels=intermediate_feats, ps=ps,
                                       tf_drop_ratio=tf_drop_ratio, img_shape=img_shape)

        self.conv1 = nn.Sequential(
            nn.Conv2d(intermediate_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        if is_lastblock is False:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(edge_feats, edge_feats // 2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(edge_feats // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_feats // 2, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, rgb_feature, edge_feature, qk1, qk2, target=None):
        if target is None:
            x = torch.cat((rgb_feature, edge_feature), dim=1)
        else:
            x = torch.cat((rgb_feature, edge_feature, target), dim=1)

        x = self.squeeze_conv(x)
        x = self.attn_fuse(feature=x, qk1=qk1, qk2=qk2)
        x = self.conv1(x)
        # print('x.shape:{}'.format(x.shape))
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.conv2(x)

        return x


class DecoderV2M1(nn.Module):
    def __init__(self):
        super(DecoderV2M1, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, 0.1, 4)]
        self.decoder_block1 = DecoderBlockV2(block_sn=1, rgb_feats=256, edge_feats=256, output_feats=128,
                                             scale_factor=2.0, ps=1, tf_drop_ratio=dpr[0],
                                             img_shape=(15, 20))
        self.decoder_block2 = DecoderBlockV2(block_sn=2, rgb_feats=128, edge_feats=128, output_feats=64,
                                             target_feats=128, scale_factor=2.0, ps=2, tf_drop_ratio=dpr[1],
                                             img_shape=(30, 40))
        self.decoder_block3 = DecoderBlockV2(block_sn=3, rgb_feats=64, edge_feats=64, output_feats=32,
                                             target_feats=64, scale_factor=2.0, ps=4, tf_drop_ratio=dpr[2],
                                             img_shape=(60, 80))
        self.decoder_block4 = DecoderBlockV2(block_sn=4, rgb_feats=32, edge_feats=32, output_feats=1,
                                             target_feats=32, scale_factor=4.0, ps=8, tf_drop_ratio=dpr[3],
                                             img_shape=(120, 160), is_lastblock=True)

    def forward(self, rgb_features, edge_features, rgb_qk_list, edge_qk_list):
        x1, x2, x3, x4 = rgb_features[0], rgb_features[3], rgb_features[1], rgb_features[2]

        x_list = []
        x_list.append(rgb_features[0])
        x_list.append(rgb_features[3])
        x_list.append(rgb_features[1])
        x_list.append(rgb_features[2])

        # for i in range(len(rgb_features)):
        #     print('rgb_features[{}].shape:{}'.format(i, rgb_features[i].shape))
        # for i in range(len(x_list)):
        #     print('x_list[{}].shape:{}'.format(i, x_list[i].shape))
        # for i in range(len(edge_features)):
        #     print('edge_features[{}].shape:{}'.format(i, edge_features[i].shape))
        # for i in range(len(rgb_qk_list)):
        #     print('rgb_qk_list[{}].shape:{}'.format(i, rgb_qk_list[i].shape))
        # for i in range(len(edge_qk_list)):
        #     print('edge_qk_list[{}].shape:{}'.format(i, edge_qk_list[i].shape))

        # y = self.decoder_block1(x4, edge_features[-1], rgb_qk_list[-1], edge_qk_list[-1])
        # y = self.decoder_block2(x3, edge_features[-2], rgb_qk_list[-2], edge_qk_list[-2], target=y)
        # y = self.decoder_block3(x2, edge_features[-3], rgb_qk_list[-3], edge_qk_list[-3], target=y)
        # y = self.decoder_block4(x1, edge_features[-4], rgb_qk_list[-4], edge_qk_list[-4], target=y)

        y = self.decoder_block1(x_list[-1], edge_features[-1], rgb_qk_list[-1], edge_qk_list[-1])
        y = self.decoder_block2(x_list[-2], edge_features[-2], rgb_qk_list[-2], edge_qk_list[-2], target=y)
        y = self.decoder_block3(x_list[-3], edge_features[-3], rgb_qk_list[-3], edge_qk_list[-3], target=y)
        y = self.decoder_block4(x_list[-4], edge_features[-4], rgb_qk_list[-4], edge_qk_list[-4], target=y)

        return y


class DecoderBlockV3(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 attn_type=None,
                 scale_factor=4.0,
                 is_lastblock=False):
        super(DecoderBlockV3, self).__init__()
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
                nn.Conv2d(edge_feats, edge_feats // 2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(edge_feats // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(edge_feats // 2, 1, kernel_size=1, stride=1, padding=0),
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


class DecoderV3(nn.Module):
    def __init__(self, attn_type=None):
        super(DecoderV3, self).__init__()

        self.decoder_block_1 = DecoderBlockV3(1, 64, 64, output_feats=64, attn_type=attn_type, scale_factor=2.0)
        self.decoder_block_2 = DecoderBlockV3(2, 64, 32, output_feats=32, target_feats=64, attn_type=attn_type,
                                              scale_factor=2.0)
        self.decoder_block_3 = DecoderBlockV3(3, 32, 16, output_feats=1, target_feats=32, attn_type=attn_type,
                                              scale_factor=2.0, is_lastblock=True)

    def forward(self, features, edges, x=None):
        # for i in range(len(features)):
        #     print('features[{}].shape:{}'.format(i, features[i].shape))
        # for i in range(len(edges)):
        #     print('edges[{}].shape:{}'.format(i, edges[i].shape))

        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[1], edges[-2], target=y)
        y = self.decoder_block_3(features[0], edges[-3], target=y)

        return y


class DecoderV3M1(nn.Module):
    def __init__(self, attn_type=None):
        super(DecoderV3M1, self).__init__()

        self.decoder_block_1 = DecoderBlockV3(1, 128, 128, output_feats=64, attn_type=attn_type, scale_factor=2.0)
        self.decoder_block_2 = DecoderBlockV3(2, 64, 64, output_feats=32, target_feats=64, attn_type=attn_type,
                                              scale_factor=2.0)
        self.decoder_block_3 = DecoderBlockV3(3, 32, 32, output_feats=1, target_feats=32, attn_type=attn_type,
                                              scale_factor=2.0, is_lastblock=True)

    def forward(self, features, edges, x=None):
        # for i in range(len(features)):
        #     print('features[{}].shape:{}'.format(i, features[i].shape))
        # for i in range(len(edges)):
        #     print('edges[{}].shape:{}'.format(i, edges[i].shape))

        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)

        return y


class DecoderV3M2(nn.Module):
    def __init__(self, attn_type=None):
        super(DecoderV3M2, self).__init__()

        self.decoder_block_1 = DecoderBlockV3(1, 64, 64, output_feats=32, attn_type=attn_type, scale_factor=2.0)
        self.decoder_block_2 = DecoderBlockV3(2, 64, 64, output_feats=32, target_feats=32, attn_type=attn_type,
                                              scale_factor=2.0)
        self.decoder_block_3 = DecoderBlockV3(3, 32, 32, output_feats=1, target_feats=32, attn_type=attn_type,
                                              scale_factor=2.0, is_lastblock=True)

    def forward(self, features, edges, x=None):
        # for i in range(len(features)):
        #     print('features[{}].shape:{}'.format(i, features[i].shape))
        # for i in range(len(edges)):
        #     print('edges[{}].shape:{}'.format(i, edges[i].shape))

        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)

        return y


class DecoderV3M3(nn.Module):
    def __init__(self, attn_type=None):
        super(DecoderV3M3, self).__init__()

        self.decoder_block_1 = DecoderBlockV3(1, 64, 64, output_feats=32, attn_type=attn_type, scale_factor=2.0)
        self.decoder_block_2 = DecoderBlockV3(2, 64, 64, output_feats=32, target_feats=32, attn_type=attn_type,
                                              scale_factor=2.0)
        self.decoder_block_3 = DecoderBlockV3(3, 32, 32, output_feats=1, target_feats=32, attn_type=attn_type,
                                              scale_factor=4.0, is_lastblock=True)

    def forward(self, features, edges, x=None):
        # for i in range(len(features)):
        #     print('features[{}].shape:{}'.format(i, features[i].shape))
        # for i in range(len(edges)):
        #     print('edges[{}].shape:{}'.format(i, edges[i].shape))

        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)

        return y


class DecoderBlockV4(nn.Module):
    def __init__(self,
                 block_sn,
                 rgb_feats,
                 edge_feats,
                 output_feats,
                 target_feats=None,
                 attn_type=None,
                 scale_factor=2.0):
        super(DecoderBlockV4, self).__init__()
        self.block_sn = block_sn
        self.attn_type = attn_type
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

        self.conv2 = nn.Sequential(
            nn.Conv2d(edge_feats, edge_feats, kernel_size=3, stride=1, padding=1),
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

        self.conv3 = nn.Sequential(
            nn.Conv2d(edge_feats, output_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(output_feats),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, rgb_feature, edge, target=None):
        if target is None:
            x = torch.cat((rgb_feature, edge), dim=1)
        else:
            x = torch.cat((rgb_feature, edge, target), dim=1)

        x = self.conv2(self.conv1(x))

        if self.attn_type is not None:
            x = self.attn(x)

        x = x + edge

        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')

        x = self.conv3(x)

        return x


class OutputHead(nn.Module):
    def __init__(self, in_c, scale_factor=2.0):
        super(OutputHead, self).__init__()

        self.scale_factor = scale_factor
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_c, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.output_conv(x)

        return x


class DecoderV4(nn.Module):
    def __init__(self, attn_type=None):
        super(DecoderV4, self).__init__()

        self.decoder_block_1 = DecoderBlockV3(1, 64, 64, output_feats=64, attn_type=attn_type, scale_factor=2.0)
        self.decoder_block_2 = DecoderBlockV3(2, 64, 64, output_feats=32, target_feats=64, attn_type=attn_type,
                                              scale_factor=2.0)
        self.decoder_block_3 = DecoderBlockV3(3, 32, 32, output_feats=16, target_feats=32, attn_type=attn_type,
                                              scale_factor=2.0)
        self.output_head = OutputHead(16, scale_factor=2.0)

    def forward(self, features, edges, x=None):
        # for i in range(len(features)):
        #     print('features[{}].shape:{}'.format(i, features[i].shape))
        # for i in range(len(edges)):
        #     print('edges[{}].shape:{}'.format(i, edges[i].shape))

        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)
        y = self.output_head(y)

        return y


# NOTE: 原版
# class DDRNetEdgeV2(nn.Module):
#     def __init__(self, opts):
#         super(DDRNetEdgeV2, self).__init__()
#
#         bs = getattr(opts, "common.bs", 8)
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#         # NOTE: DualResNet_Backbone+EdgeNet Params(3.63M)  完全结构的DDRNet参数5.70M
#         # 原始trim版本
#         # [b, 32, 120, 160]  ✔
#         # [b, 128, 30, 40]   ✔
#         # [b, 256, 15, 20]   ✔
#         # [b, 128, 60, 80]   ✔ 最后一层需要与第二层调换顺序使用
#         # 原始23-slim版本
#         # [B, 32, 120, 160]  !  layer1
#         # [B, 64, 60, 80]    !  layer2
#         # [B, 128, 30, 40]      layer3
#         # [B, 256, 15, 20]      layer4
#         # [B, 64, 60, 80]    !  final_layer
#         self.rgb_feature_extractor = DualResNet_Backbone(opts, pretrained=True, features=64)
#         self.bottleneck_conv = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#             # nn.BatchNorm2d(64),
#             nn.Sigmoid()
#         )
#         self.bottleneck_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#
#         # r = 1
#         # self.bottleneck_conv1 = nn.Conv2d(32, 32 // r, kernel_size=1, stride=1, padding=0)
#         # self.bottleneck_conv2 = nn.Conv2d(128, 64 // r, kernel_size=1, stride=1, padding=0)
#         # self.bottleneck_conv3 = nn.Conv2d(128, 128 // r, kernel_size=1, stride=1, padding=0)
#         # self.bottleneck_conv4 = nn.Conv2d(256, 256 // r, kernel_size=1, stride=1, padding=0)
#
#         # self.rgb_feature_mhsa = BackboneMultiScaleAttnExtractor1()
#         # self.rgb_feature_mhsa = BackboneMultiScaleAttnExtractor2()
#         self.rgb_feature_mhsa = BackboneMultiScaleAttnExtractor3()
#
#         # self.edge_feature_extractor = EdgeNetV1(reduction=1, img_shape=(480, 640))  # NOTE: Params(0.25M)
#         # self.edge_feature_extractor = EdgeNetV2(reduction=1, img_shape=(480, 640))
#         # self.edge_feature_extractor = EdgeNetV1M(reduction=1, img_shape=(480, 640))
#         # self.edge_feature_extractor = EdgeNetV3(reduction=1, img_shape=(480, 640))
#         # self.edge_feature_extractor = EdgeNetV4(reduction=1, img_shape=(480, 640))
#         self.edge_feature_extractor = EdgeNetV5(reduction=1, img_shape=(480, 640))
#         # self.edge_feature_extractor = EdgeNetV6(reduction=1)
#         # self.edge_feature_extractor = EdgeNetV5Attn(reduction=1)
#
#         # self.decoder = DecoderV1(attn_type='cbam')  # NOTE: Params(0.26M)
#         # self.decoder = DecoderV2()
#         # self.decoder = DecoderV2M1()
#         # self.decoder = DecoderV3(attn_type='cbam')
#         # self.decoder = DecoderV3M1(attn_type='cbam')
#         self.decoder = DecoderV3M2(attn_type='cbam')
#         # self.decoder = DecoderV4(attn_type='cbam')
#         # self.decoder = DecoderV3M3(attn_type='cbam')
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         # NOTE: 提取RGB的context特征信息
#         # t0 = time.time()
#         rgb_features = self.rgb_feature_extractor(x)  # [b, 64, 60, 80]
#         # for i in range(len(rgb_features)):
#         #     print('rgb_features[{}].shape:{}'.format(i, rgb_features[i].shape))
#         # print("features, max:{}, min:{}".format(features[-1].max(), features[-1].min()))
#         # print(features[-1])
#         # print("rgb_feature_extractor time:{:.4f}s".format(time.time() - t0))
#
#         # rgb_features[-1] = self.bottleneck_conv2(rgb_features[-1])
#         # rgb_features[-1] = self.bottleneck_conv(rgb_features[-1])
#         # rgb_features[-1] = self.bottleneck_pool(rgb_features[-1])
#         # for i in range(len(rgb_features)):
#         #     print('rgb_features[{}].shape:{}'.format(i, rgb_features[i].shape))
#
#         # rgb_features, similarity_qk_list = self.rgb_feature_mhsa(rgb_features)
#         rgb_features = self.rgb_feature_mhsa(rgb_features)
#         # rgb_similarity_qk_list = self.rgb_feature_mhsa(rgb_features)
#         # for i in range(len(rgb_features)):
#         #     print('rgb_features[{}].shape:{}'.format(i, rgb_features[i].shape))
#
#         # NOTE: 提取RGB的edge
#         # t0 = time.time()
#         x_edge = self.edge_extractor_torch(x, device=self.device)
#         # print("edge_extractor time:{:.6f}s".format(time.time() - t0))
#
#         # NOTE: 提取edge特征信息
#         # t0 = time.time()
#         edge_features = self.edge_feature_extractor(x_edge)
#         # edge_features, edge_similarity_qk_list = self.edge_feature_extractor(x_edge)
#         # for i in range(len(edge_features)):
#         #     print('edge_features[{}].shape:{}'.format(i, edge_features[i].shape))
#         # exit()
#         # print("edges, max:{}, min:{}".
#         # format(np.max(edges[-1].detach().cpu().numpy()), np.min(edges[-1].detach().cpu().numpy())))
#         # print(edges[-1])
#         # print("edge_feature_extractor time:{:.6f}s".format(time.time() - t0))
#
#         # NOTE: 上采样稠密深度估计
#         # t0 = time.time()
#         y = self.decoder(rgb_features, edge_features)
#         # y = self.decoder(rgb_features, edge_features, rgb_similarity_qk_list, edge_similarity_qk_list)
#         # print("decoder time:{:.4f}s".format(time.time() - t0))
#
#         # NOTE: 可视化
#         # x_ = x
#         # x_ = x_ * 255
#         # x_ = x_.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
#         # b, _, _, _ = x.size()
#         # print("b:", b)
#         # x_edge = x_edge.detach().cpu().numpy().astype(np.uint8)
#         # x_edge = x_edge.transpose(0, 2, 3, 1)
#         #
#         # cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
#         # cmap_type = cmap_type_list[1]
#         # for i in range(b):
#         #     plt.subplot(2, 4, i + 1)
#         #     plt.imshow(x_[i])  #
#         #     plt.subplot(2, 4, i + 5)
#         #     plt.imshow(x_edge[i], cmap=cmap_type)
#         # plt.show()
#
#         return y
#
#     def edge_extractor_torch(self, x: Tensor, device):
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


class SGNetV2(nn.Module):
    def __init__(self, out_feats=[64, 64, 32], intermediate_feat=96):
        super(SGNetV2, self).__init__()
        self.out_feats = out_feats

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
        x1 = F.interpolate(x, scale_factor=0.125, mode='bilinear')  # 1/8
        x2 = F.interpolate(x, scale_factor=0.125, mode='bilinear')  # 1/8
        x3 = F.interpolate(x, scale_factor=0.25, mode='bilinear')  # 1/4

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


class SGNetV3(nn.Module):
    def __init__(self, out_feats=[64, 48, 32], intermediate_feat=96):
        super(SGNetV3, self).__init__()
        self.out_feats = out_feats

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
        x1 = F.interpolate(x, scale_factor=0.125, mode='bilinear')  # 1/8
        x2 = F.interpolate(x, scale_factor=0.250, mode='bilinear')  # 1/4
        x3 = F.interpolate(x, scale_factor=0.500, mode='bilinear')  # 1/2

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


class OutputHeadV4M1(nn.Module):
    def __init__(self,
                 in_channels=[64, 64, 32],
                 out_channels=64):
        super(OutputHeadV4M1, self).__init__()

        # self.bot_conv = nn.Conv2d(
        #     in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        # self.skip_conv1 = nn.Conv2d(
        #     in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)  # 32 out_channels

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        max_res_fm_channels = 32  # 最大分辨率的特征图的通道数
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)  # out_channels
        # self.fusion3 = SelectiveFeatureFusion(max_res_fm_channels)  # out_channels
        # self.squeeze_conv = nn.Conv2d(out_channels, max_res_fm_channels, kernel_size=1, stride=1, padding=0)
        self.transit_conv = nn.Conv2d(out_channels, max_res_fm_channels, kernel_size=3, stride=1, padding=1)

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(max_res_fm_channels, max_res_fm_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max_res_fm_channels, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, features):
        # for i in range(len(features)):
        #     print('features[{}].shape: {}'.format(i, features[i].shape))

        out = self.fusion1(features[0], features[1])  # [b, 64, h/8, w/8]
        out = self.up(out)  # [b, 64, h/4, w/4]

        x_2_ = self.skip_conv2(features[2])
        out = self.fusion2(x_2_, out)  # [b, 64, h/4, w/4]
        out = self.up(out)  # [b, 64, h/2, w/2]

        out = self.transit_conv(out)  # [b, 32, h/2, w/2]
        out = self.up(out)  # [b, 32, h/1, w/1]
        # print('out.shape:{}'.format(out.shape))

        out = self.last_layer_depth(out)

        return out


class CMTFDecoderBlockV1(nn.Module):
    def __init__(self,
                 in_feats=64,
                 out_feats=48,
                 cm_tf_block_nums=1,
                 basic_tf_block_nums=3,
                 drop_path_ratio=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6,
                 alpha='learnable',
                 key_dim=16,
                 num_heads=8,
                 mlp_ratio=2,
                 attn_ratio=2,
                 drop=0,
                 fm_res=8,
                 ):
        super(CMTFDecoderBlockV1, self).__init__()

        if alpha == 'learnable':
            self.alpha = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        else:
            self.alpha = alpha

        self.cm_tf_block_nums = cm_tf_block_nums
        self.basic_tf_block_nums = basic_tf_block_nums
        self.cm_tf_transformer_blocks = nn.ModuleList()
        self.basic_tf_transformer_blocks = nn.ModuleList()

        for i in range(cm_tf_block_nums):
            self.cm_tf_transformer_blocks.append(
                CrossModalTFBlockV3M1(
                    dim=in_feats,
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,
                    drop_path=drop_path_ratio,
                    norm_cfg=norm_cfg,
                    act_layer=act_layer,
                    alpha=self.alpha,
                    fm_res=fm_res  # feature map resolution: 1/fm_res
                )
            )
        # self.cm_tf_transformer_block = CrossModalTFBlockV3M1(dim=in_feats,
        #                                                      key_dim=key_dim,  # 16
        #                                                      num_heads=num_heads,  # 8
        #                                                      mlp_ratio=mlp_ratio,  # 2
        #                                                      attn_ratio=attn_ratio,  # 2
        #                                                      drop=drop,
        #                                                      drop_path=drop_path_ratio,
        #                                                      norm_cfg=norm_cfg,
        #                                                      act_layer=act_layer,
        #                                                      alpha=self.alpha,
        #                                                      fm_res=fm_res  # feature map resolution: 1/fm_res
        #                                                      )

        for i in range(basic_tf_block_nums):
            self.basic_tf_transformer_blocks.append(
                TFBlock(
                    dim=in_feats,
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    drop_path=drop_path_ratio,
                    norm_cfg=norm_cfg,
                    act_layer=act_layer
                )
            )

        intermediate_feat = 2 * in_feats
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feats, intermediate_feat, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_feat),
            # nn.ReLU6(inplace=True),
            nn.ReLU(inplace=True),
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(intermediate_feat, out_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feats),
            # nn.ReLU6(inplace=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, guide):
        B, C, H, W = get_shape(rgb)
        _, c, h, w = get_shape(guide)
        if C != c or H != h or W != w:
            print('RGB embedding should have the same shape with guide embedding.')
            exit()

        # print('rgb.shape:{}'.format(rgb.shape))
        # print('guide.shape:{}'.format(guide.shape))
        shortcut_rgb_feat = rgb
        shortcut_guide_feat = guide
        x = self.cm_tf_transformer_blocks[0](rgb, guide)
        for i in range(self.basic_tf_block_nums):
            x = self.basic_tf_transformer_blocks[i](x)
        # print('x.shape:{}'.format(x.shape))
        # exit()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)  # 直接双边采样从1/64恢复到送进来时的特征图尺寸
        x = x + shortcut_rgb_feat + shortcut_guide_feat

        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)

        return x


class CMTFDecoderV1(nn.Module):
    def __init__(self):
        super(CMTFDecoderV1, self).__init__()

        # self.block1 = CMTFDecoderBlockV1(in_feats=64, out_feats=48)
        # self.block2 = CMTFDecoderBlockV1(in_feats=48, out_feats=32)
        # self.block3 = CMTFDecoderBlockV1(in_feats=32, out_feats=1)
        self.block1 = CMTFDecoderBlockV1(in_feats=64, out_feats=48, act_layer=nn.ReLU)
        self.block2 = CMTFDecoderBlockV1(in_feats=48, out_feats=32, act_layer=nn.ReLU)
        self.block3 = CMTFDecoderBlockV1(in_feats=32, out_feats=1, act_layer=nn.ReLU)

    def forward(self, rgb_feat, guide_feats):
        # for i in range(len(guide_feats)):
        #     print('guide_feats[{}].shape:{}'.format(i, guide_feats[i].shape))
        x = self.block1(rgb_feat, guide_feats[0])
        # print('x.shape:{}'.format(x.shape))
        x = self.block2(x, guide_feats[1])
        x = self.block3(x, guide_feats[2])
        x = 80.0 * x

        return x


class DDRNetEdgeV2(nn.Module):
    def __init__(self, opts):
        super(DDRNetEdgeV2, self).__init__()

        bs = getattr(opts, "common.bs", 8)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # NOTE: DualResNet_Backbone+EdgeNet Params(3.63M)  完全结构的DDRNet参数5.70M
        # trim5
        # [b, 32, 120, 160]  ✔
        # [b, 64, 60, 80]   ✔
        # [b, 128, 30, 40]   ✔
        # [b, 256, 15, 20]   ✔
        # trim6
        # [b, 32, 120, 160]  ✔
        # [b, 64, 60, 80]   ✔
        # [b, 128, 30, 40]   ✔
        # [b, 256, 15, 20]   ✔
        self.rgb_feature_extractor = DualResNet_Backbone(opts, pretrained=True, features=64)
        # self.bottleneck_conv = nn.Conv2d(256, 160, kernel_size=1, stride=1, padding=0)

        # NOTE: DDRNet_trim7_ENV4_TMV3M4_LearnableAlpha_OHV4_09-23
        # self.bottleneck_conv = nn.Conv2d(64, 160, kernel_size=1, stride=1, padding=0)
        # self.edge_feature_extractor = ENV4(reduction=1)
        # self.transition_module = TransitionModuleV3M4(feats_list=[64, 128, 160], alpha='learnable')
        # self.decoder = OutputHeadV4(in_channels=[160, 128, 64], out_channels=64)

        # NOTE: DDRNet-slim_SGN_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-09
        # self.guidance_feature_extractor = SGNet(out_feats=[64, 64, 64])  # NOTE: Params(0.09M)
        # self.transition_module = TransitionModuleV3M4(feats_list=[64, 64, 64], alpha='learnable')
        # self.decoder = OutputHeadV4(in_channels=[64, 64, 64], out_channels=64)

        # NOTE: DDRNet-slim_SGNV2_TMV3M4_LA_OHV4M1_kitti-alhashim_10-09
        # self.guidance_feature_extractor = SGNetV2(out_feats=[64, 64, 32])
        # self.transition_module = TransitionModuleV3M4(feats_list=[32, 64, 64], alpha='learnable')
        # self.decoder = OutputHeadV4M1(in_channels=[64, 64, 32], out_channels=64)

        # NOTE: DDRNet-slim_SGNV3_CMTFDecoderV1_10-13
        #       DDRNet-slim_SGNV3_CMTFDecoderV1_kitti-alhashim_10-13
        self.guidance_feature_extractor = SGNetV3(out_feats=[64, 48, 32])
        self.decoder = CMTFDecoderV1()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, speed_test=False):
        # NOTE: DDRNet_trim5_ENV4_TMV3M4_LearnableAlpha_OHV4_09-16
        # rgb_features = self.rgb_feature_extractor(x)  # NOTE: 提取RGB的context特征信息
        # rgb_features[-1] = self.bottleneck_conv(rgb_features[-1])
        # rgb_features_ = rgb_features[::-1][:3]  # 倒序取出每层特征 并只取最后三层
        # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
        # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
        # fused_features = self.transition_module(rgb_features_, edge_features)  # NOTE: 融合RGB和edge信息过渡给decoder
        # fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
        # y = self.decoder(fused_features)

        # # NOTE: DDRNet_slim_ENV4_TMV3M4_LearnableAlpha_OHV4_09-16
        # rgb_features = self.rgb_feature_extractor(x)  # NOTE: 提取RGB的context特征信息
        # # for i in range(len(rgb_features)):
        # #     print('rgb_features[{}].shape:{}'.format(i, rgb_features[i].shape))
        # # exit()
        # rgb_features_ = []
        # rgb_features_.append(rgb_features[3])  # 1/32
        # rgb_features_.append(rgb_features[2])  # 1/16
        # rgb_features_.append(rgb_features[-1])  # 1/8
        # rgb_features_[0] = self.bottleneck_conv(rgb_features_[0])
        # # for i in range(len(rgb_features_)):
        # #     print('rgb_features_[{}].shape:{}'.format(i, rgb_features_[i].shape))
        # # exit()
        # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
        # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
        # fused_features = self.transition_module(rgb_features_, edge_features)  # NOTE: 融合RGB和edge信息过渡给decoder
        # fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
        # y = self.decoder(fused_features)

        # # NOTE: DDRNet_trim6_ENV4_TMV3M4_LearnableAlpha_OHV4_09-23
        # rgb_features = self.rgb_feature_extractor(x)  # NOTE: 提取RGB的context特征信息
        # rgb_features_ = rgb_features[1:][::-1]  # 取出1/8 1/16 1/32三种并倒序排列
        # rgb_features_[0] = self.bottleneck_conv(rgb_features_[0])
        # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
        # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
        # fused_features = self.transition_module(rgb_features_, edge_features)  # NOTE: 融合RGB和edge信息过渡给decoder
        # fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
        # y = self.decoder(fused_features)

        # NOTE: DDRNet_trim7_ENV4_TMV3M4_LearnableAlpha_OHV4_09-23
        # rgb_features = self.rgb_feature_extractor(x)  # NOTE: 提取RGB的context特征信息
        # # for i in range(len(rgb_features)):
        # #     print('rgb_features[{}].shape:{}'.format(i, rgb_features[i].shape))
        # # exit()
        # rgb_features_ = rgb_features[1:][::-1]  # 取出1/8 1/16 1/32三种并倒序排列
        # rgb_features_[0] = self.bottleneck_conv(rgb_features_[0])  # 64 channels -> 160 channels
        # _, _, h, w = rgb_features_[0].shape
        # rgb_features_[0] = nn.functional.adaptive_avg_pool2d(rgb_features_[0], (h // 4, w // 4))
        # x_edge = self.edge_extractor_torch(x, device=self.device)  # NOTE: 提取RGB的edge
        # edge_features = self.edge_feature_extractor(x_edge)  # NOTE: 提取edge的特征信息
        # fused_features = self.transition_module(rgb_features_, edge_features)  # NOTE: 融合RGB和edge信息过渡给decoder
        # fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
        # y = self.decoder(fused_features)

        # NOTE: DDRNet-slim_SGN_TMV3M4_LearnableAlpha_OHV4_kitti-alhashim_10-09
        # rgb_features = self.rgb_feature_extractor(x)  # NOTE: 提取RGB的context特征信息
        # rgb_features = rgb_features[:2] + rgb_features[-1:]  # 取出4 8 8前两层和最后一层特征图
        # rgb_features.append(rgb_features[-1])
        # _, _, h, w = x.shape  # h, w
        # rgb_features[-2] = nn.functional.adaptive_avg_pool2d(rgb_features[-2], (h // 16, w // 16))
        # rgb_features[-1] = nn.functional.adaptive_avg_pool2d(rgb_features[-1], (h // 32, w // 32))
        # guidance_features = self.guidance_feature_extractor(x)
        # rgb_features_ = rgb_features[::-1][:3]  # 倒序取出每层特征 并只取最后三层
        # fused_features = self.transition_module(rgb_features_, guidance_features)
        # fused_features.append(rgb_features[0])  # 把1/4尺度的特征图添加进去
        # y = self.decoder(fused_features)

        # NOTE: DDRNet-slim_SGNV2_TMV3M4_LA_OHV4M1_kitti-alhashim_10-09
        # rgb_features = self.rgb_feature_extractor(x)
        # rgb_features = rgb_features[:2] + rgb_features[-1:]  # 取出4 8 8前两层和最后一层特征图
        # rgb_features = rgb_features[::-1]  # 对特征进行倒序
        # guidance_features = self.guidance_feature_extractor(x)
        # fused_features = self.transition_module(rgb_features, guidance_features)
        # y = self.decoder(fused_features)

        # NOTE: DDRNet-slim_SGNV2_CMTFDecoderV1_10-13
        #       DDRNet-slim_SGNV3_CMTFDecoderV1_kitti-alhashim_10-13
        rgb_features = self.rgb_feature_extractor(x)
        rgb_feature = rgb_features[-1]
        guidance_features = self.guidance_feature_extractor(x)
        y = self.decoder(rgb_feature, guidance_features)

        return y

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
