import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os

# try:
#     from mmdet.models.builder import BACKBONES as det_BACKBONES
#     from mmdet.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmdet = True
# except ImportError:
#     print("If for detection, please install mmdetection first")
#     has_mmdet = False

import torch.nn.functional as F

from utils import logger
from ..modules import PConvGuidedUpsampleBlock
import time


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
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
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


Embedding_Channels = {"T1": 64, "T2": 96, "S": 128}
depth_dicts = {"T2": (1, 2, 8, 2), 'S': (1, 2, 13, 2)}


class FasterNetPCGUB(nn.Module):
    def __init__(self,
                 opts,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
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
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()
        ''' Encoder '''
        mode = getattr(opts, "model.mode", "T2")
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

        # TODO:这里的卷积也能替换成PConv
        if mode == "T2":
            self.btlnck_conv = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)  # btlnck conv
        elif mode == 'S':
            self.btlnck_conv = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # btlnck conv

        ''' Decoder GUB'''
        up_features = [256, 128, 64, 32, 16]
        expand_features = [256, 128, 64, 32, 16]
        self.decoder_up1 = PConvGuidedUpsampleBlock(in_features=up_features[0],
                                                   expand_features=expand_features[0],
                                                   out_features=up_features[1],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 80, 30, 40]

        self.decoder_up2 = PConvGuidedUpsampleBlock(in_features=up_features[1],
                                                   expand_features=expand_features[1],
                                                   out_features=up_features[2],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 40, 60, 80]

        self.decoder_up3 = PConvGuidedUpsampleBlock(in_features=up_features[2],
                                                   expand_features=expand_features[2],
                                                   out_features=up_features[3],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 20, 120, 160]

        self.decoder_up4 = PConvGuidedUpsampleBlock(in_features=up_features[3],
                                                   expand_features=expand_features[3],
                                                   out_features=up_features[4],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 10, 240, 320]

        self.decoder_up5 = PConvGuidedUpsampleBlock(in_features=up_features[4],
                                                   expand_features=expand_features[4],
                                                   out_features=1,
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 1, 480, 640]

        # self.apply(self.cls_init_weights)
        # self.init_cfg = copy.deepcopy(init_cfg)
        # if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
        #     self.init_weights()

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

    # init for mmdetection by loading imagenet pre-trained weights
    # def init_weights(self, pretrained=None):
    #     logger = get_root_logger()
    #     if self.init_cfg is None and pretrained is None:
    #         logger.warn(f'No pre-trained weights for '
    #                     f'{self.__class__.__name__}, '
    #                     f'training start from scratch')
    #         pass
    #     else:
    #         assert 'checkpoint' in self.init_cfg, f'Only support ' \
    #                                               f'specify `Pretrained` in ' \
    #                                               f'`init_cfg` in ' \
    #                                               f'{self.__class__.__name__} '
    #         if self.init_cfg is not None:
    #             ckpt_path = self.init_cfg['checkpoint']
    #         elif pretrained is not None:
    #             ckpt_path = pretrained
    #
    #         ckpt = _load_checkpoint(
    #             ckpt_path, logger=logger, map_location='cpu')
    #         if 'state_dict' in ckpt:
    #             _state_dict = ckpt['state_dict']
    #         elif 'model' in ckpt:
    #             _state_dict = ckpt['model']
    #         else:
    #             _state_dict = ckpt
    #
    #         state_dict = _state_dict
    #         missing_keys, unexpected_keys = \
    #             self.load_state_dict(state_dict, False)
    #
    #         # show for debug
    #         print('missing_keys: ', missing_keys)
    #         print('unexpected_keys: ', unexpected_keys)

    def decoder_upsample(self, x: Tensor, feature: Tensor, *args, **kwargs):
        y = feature
        features_ = []
        guide_maps = [x]  # [,,480, 640] [,,240, 320] [,,120, 160] [,,60, 80] [,,30, 40]
        for i in range(4):
            guide_maps.append(F.interpolate(x, scale_factor=(0.5 ** (i+1))))
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 256, 30, 40]
        y = self.decoder_up1(guide_maps[-1], y)  # [B, 128, 30, 40]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 128, 60, 80]
        y = self.decoder_up2(guide_maps[-2], y)  # [B, 64, 60, 80]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 64, 120, 160]
        y = self.decoder_up3(guide_maps[-3], y)  # [B, 32, 120, 160]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 32, 240, 320]
        y = self.decoder_up4(guide_maps[-4], y)  # [B, 16, 240, 320]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 16, 480, 640]
        y = self.decoder_up5(guide_maps[-5], y)  # [B, 1, 480, 640]
        features_.append(y)
        return y  # features_

    def forward(self, x):
        t0 = time.time()
        shortcut = x
        # NOTE: extract features
        x = self.patch_embed(x)
        x = self.stages(x)  # T2 mode:[B, 768, 15, 20]
        # print("x.shape:\n", x.shape)
        print("rgb_feature_extractor time:{:.4f}s".format(time.time() - t0))

        t0 = time.time()
        x = self.btlnck_conv(x)  # [B, 256, 15, 20]
        # NOTE: decoder upsample
        res = self.decoder_upsample(shortcut, x)
        print("5 layers PCGUB time:{:.4f}s".format(time.time() - t0))
        # for i in range(len(res)):
        #     print("decoder_{}:{}".format(i, res[i].shape))

        return res  # x, res

    def forward_cls(self, x):
        # output only the features of last layer for image classification
        x = self.patch_embed(x)
        x = self.stages(x)  # T2 mode:[B, 768, 15, 20]
        print("x.shape:\n", x.shape)
        x = self.btlnck_conv(x)
        print("x.shape:\n", x.shape)
        # x = self.avgpool_pre_head(x)  # B C 1 1
        # x = torch.flatten(x, 1)
        # x = self.head(x)

        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return outs