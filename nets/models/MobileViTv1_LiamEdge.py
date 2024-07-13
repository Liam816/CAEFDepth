import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn
import argparse
from typing import Dict, Tuple, Optional, Union, Any
from utils import logger

from ..layers import ConvLayer
from ..modules import InvertedResidual, MobileViTBlock, Guided_Upsampling_Block
from ..misc.common import parameter_list
from ..modules.cbam import CBAM
from ..modules.se import SELayer
from ..modules.eca_module import eca_layer
from ..modules.srm_module import SRMLayer
from ..modules.gct_module import GCTLayer
import sys

from typing import Dict
import numpy as np
import cv2 as cv


def get_configuration(opts) -> Dict:
    mode = getattr(opts, "model.mit.mode", "small")
    if mode is None:
        logger.error("Please specify mode")

    head_dim = getattr(opts, "model.mit.head_dim", None)  # None
    num_heads = getattr(opts, "model.mit.number_heads", 4)  # 4

    # logger.info("LIAM head_dim:{}".format(head_dim))
    # logger.info("LIAM num_heads:{}".format(num_heads))

    if head_dim is not None:
        if num_heads is not None:
            logger.error(
                "--model.mit.head-dim and --model.mit.number-heads "
                "are mutually exclusive."
            )
    elif num_heads is not None:
        if head_dim is not None:
            logger.error(
                "--model.mit.head-dim and --model.mit.number-heads "
                "are mutually exclusive."
            )

    mode = mode.lower()
    if mode == "xx_small":
        mv2_exp_mult = 2
        config = {
            "layer1": {
                "out_channels": 16,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 48,
                "transformer_channels": 64,
                "ffn_dim": 128,
                "transformer_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 64,
                "transformer_channels": 80,
                "ffn_dim": 160,
                "transformer_blocks": 4,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 80,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
        }
    elif mode == "x_small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 48,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 64,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 80,
                "transformer_channels": 120,
                "ffn_dim": 240,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
        }
    elif mode == "small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 128,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 160,
                "transformer_channels": 240,
                "ffn_dim": 480,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
        }
    else:
        raise NotImplementedError

    return config


class UpSample(nn.Sequential):
    """
    上采样过程参考该文章 "High Quality Monocular Depth Estimation via Transfer Learning"
    """

    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        # self.leakyreluA = nn.LeakyReLU(0.2)
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)
        # print('class UpSample initializing.')

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear',
                             align_corners=True)
        return self.leakyreluB(self.convB(
            self.convA(torch.cat([up_x, concat_with], dim=1))))  # 论文中appendix部分提到：只对convB的结果进行leakyrelu，而不对convA进行


def normalize2img_tensor(x: Tensor):
    min_val = x.min()
    max_val = x.max()
    res = (x - min_val) / (max_val - min_val)
    res = res * 255.
    return res


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


class EdgeNetV1(nn.Module):
    def __init__(self):
        super(EdgeNetV1, self).__init__()
        # Params: 0.25M
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
            nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=1),  # kernel: [128, 64, 3, 3]
            nn.BatchNorm2d(80),
            nn.Conv2d(80, 96, kernel_size=3, stride=2, padding=1),  # kernel: [128, 128, 3, 3]
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )  # [b, 96, h/8, w/8]

    def forward(self, x):
        features = []
        x = self.edge_encoder_1(x)  # [B, 32, 240, 320]
        features.append(x)
        x = self.edge_encoder_2(x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.edge_encoder_3(x)  # [B, 96, 60, 80]
        features.append(x)
        return features


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
        add_feats = rgb_feats

        if target_feats is None:
            n_feats = rgb_feats + edge_feats
        else:
            n_feats = rgb_feats + edge_feats + target_feats

        if use_cbam is True:
            self.cbam = CBAM(n_feats)

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feats, add_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(add_feats),
            nn.ReLU(inplace=True),
        )
        self.SE_block = SELayer(add_feats, reduction=1)

        if is_lastblock is False:
            self.conv2 = nn.Sequential(
                nn.Conv2d(add_feats, output_feats, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_feats),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(add_feats, add_feats//2, kernel_size=3, stride=1, padding=1),  # 过渡卷积
                nn.BatchNorm2d(add_feats//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(add_feats//2, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, rgb_feature, edge, target=None):
        # if self.block_sn >= 2:
        #     rgb_feature = F.interpolate(rgb_feature, scale_factor=2.0, mode='bilinear')

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


class DecoderV1(nn.Module):
    def __init__(self):
        super(DecoderV1, self).__init__()

        self.decoder_block_1 = DecoderBlockV1(1, 96, 96, output_feats=64)
        self.decoder_block_2 = DecoderBlockV1(2, 64, 64, output_feats=32, target_feats=64)
        self.decoder_block_3 = DecoderBlockV1(3, 32, 32, output_feats=1, target_feats=32, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[1], edges[-2], target=y)
        y = self.decoder_block_3(features[0], edges[-3], target=y)

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


class DecoderV6(nn.Module):
    def __init__(self, attn_type='cbam'):
        super(DecoderV6, self).__init__()

        self.decoder_block_1 = DecoderBlockV6(1, 96, 96, 64, scale_factor=2.0, attn_type=attn_type)
        self.decoder_block_2 = DecoderBlockV6(2, 64, 64, 32, 64, scale_factor=2.0, attn_type=attn_type)
        self.decoder_block_3 = DecoderBlockV6(3, 32, 32, 1, 32, scale_factor=2.0, attn_type=attn_type, is_lastblock=True)

    def forward(self, features, edges, x=None):
        y = self.decoder_block_1(features[-1], edges[-1])
        y = self.decoder_block_2(features[-2], edges[-2], target=y)
        y = self.decoder_block_3(features[-3], edges[-3], target=y)

        return y


class MobileViTv1Edge(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    def __init__(self, opts, bottleneck_num_features=640, decoder_width=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        image_channels = 3
        out_channels = 16

        mobilevit_config = get_configuration(opts=opts)  # 5层layers的参数字典

        self.dilation = 1
        output_stride = kwargs.get("output_stride", None)  # 默认为None
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True

        self.gradient_checkpointing = getattr(opts, "model.gradient_checkpointing", False)  # 默认为False

        # store model configuration in a dictionary
        self.model_conf_dict = dict()

        # NOTE: MiTv1(3 layers) Params(0.75M) | MiTv1(5 layers) Params(4.83M)
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )  # 连接了batch_norm和激活函数层

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        # in_channels = out_channels
        # self.layer_4, out_channels = self._make_layer(
        #     opts=opts,
        #     input_channel=in_channels,
        #     cfg=mobilevit_config["layer4"],
        #     dilate=self.dilate_l4,
        # )
        # self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}
        #
        # in_channels = out_channels
        # self.layer_5, out_channels = self._make_layer(
        #     opts=opts,
        #     input_channel=in_channels,
        #     cfg=mobilevit_config["layer5"],
        #     dilate=self.dilate_l5,
        # )
        # self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        # NOTE: EdgeNet Params(0.16M)
        self.edge_feature_extractor = EdgeNetV1()

        # NOTE: DecoderV1 Params(0.42M)
        # self.decoder = DecoderV1()
        self.decoder = DecoderV6(attn_type='cbam')

    def forward_layer(self, layer: nn.Module, x: Tensor) -> Tensor:
        # Larger models with large input image size may not be able to fit into memory.
        # We can use gradient checkpointing to enable training with large models and large inputs
        return (
            gradient_checkpoint_fn(layer, x)
            if self.gradient_checkpointing
            else layer(x)
        )

    def rgb_feature_extractor(self, x: Tensor):
        features = []
        x = self.forward_layer(self.conv_1, x)  # [B, 16, 240, 320]
        # features.append(x)
        x = self.forward_layer(self.layer_1, x)  # [B, 32, 240, 320]
        features.append(x)
        x = self.forward_layer(self.layer_2, x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.forward_layer(self.layer_3, x)  # [B, 96, 60, 80]
        features.append(x)
        # x = self.forward_layer(self.layer_4, x)  # [B, 128, 30, 40]
        # features.append(x)
        # x = self.forward_layer(self.layer_5, x)  # [B, 160, 15, 20]
        # features.append(x)
        return features

    def forward(self, x: Any, *args, **kwargs) -> Any:
        # NOTE: 提取RGB的context特征信息
        # [B, 32, 240, 320]  !  layer1
        # [B, 64, 120, 160]  !  layer2
        # [B, 96, 60, 80]    !  layer3
        # [B, 128, 30, 40]      layer4
        # [B, 160, 15, 20]      layer5
        rgb_features = self.rgb_feature_extractor(x)
        # print("features, max:{}, min:{}".format(features[-1].max(), features[-1].min()))
        # print(features[-1])

        # NOTE: 提取RGB的edge
        # x_edge = edge_extractor(x, 'sobel')
        x_edge = self.edge_extractor_torch(x, device=self.device)

        # NOTE: 提取edge特征信息
        # [B, 32, 240, 320]  !
        # [B, 64, 120, 160]  !
        # [B, 96, 60, 80]    !
        edge_features = self.edge_feature_extractor(x_edge)
        # for i in range(len(edges)):
        #     print('edges[{}].shape:{}'.format(i, edges[i].shape))
        # print("edges, max:{}, min:{}".
        # format(np.max(edges[-1].detach().cpu().numpy()), np.min(edges[-1].detach().cpu().numpy())))
        # print(edges[-1])

        # NOTE: 上采样稠密深度估计
        # y, before_add, after_add = self.decoder(rgb_features, edges, x)
        y = self.decoder(rgb_features, edge_features)
        # y = self.decoder(rgb_features)

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
        return y  # y rgb_features

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
        sobel_xy = normalize2img_tensor(sobel_xy)

        return sobel_xy  # sobel_x, sobel_y, sobel_xy


    def get_trainable_parameters(
            self,
            weight_decay: Optional[float] = 0.0,
            no_decay_bn_filter_bias: Optional[bool] = False,
            *args,
            **kwargs
    ):
        """Get trainable parameters"""
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            *args,
            **kwargs
        )  # param_list是包含一个字典的列表，长度为1
        return param_list, [1.0] * len(param_list)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.mit.mode",
            type=str,
            default="small",
            choices=["xx_small", "x_small", "small"],
            help="MobileViT mode. Defaults to small",
        )
        group.add_argument(
            "--model.mit.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.mit.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between FFN layers. Defaults to 0.0",
        )
        group.add_argument(
            "--model.mit.dropout",
            type=float,
            default=0.0,
            help="Dropout in Transformer layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.mit.transformer-norm-layer",
            type=str,
            default="layer_norm",
            help="Normalization layer in transformer. Defaults to LayerNorm",
        )
        group.add_argument(
            "--model.mit.no-fuse-local-global-features",
            action="store_true",
            help="Do not combine local and global features in MobileViT block",
        )
        group.add_argument(
            "--model.mit.conv-kernel-size",
            type=int,
            default=3,
            help="Kernel size of Conv layers in MobileViT block",
        )

        group.add_argument(
            "--model.mit.head-dim",
            type=int,
            default=None,
            help="Head dimension in transformer",
        )
        group.add_argument(
            "--model.mit.number-heads",
            type=int,
            default=None,
            help="Number of heads in transformer",
        )
        return parser

    def _make_layer(
            self,
            opts,
            input_channel,
            cfg: Dict,
            dilate: Optional[bool] = False,
            *args,
            **kwargs
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts, input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
            opts, input_channel: int, cfg: Dict, *args, **kwargs
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(
            self,
            opts,
            input_channel,
            cfg: Dict,
            dilate: Optional[bool] = False,
            *args,
            **kwargs
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            logger.error(
                "Transformer input dimension should be divisible by head dimension. "
                "Got {} and {}.".format(transformer_dim, head_dim)
            )

        block.append(
            MobileViTBlock(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(opts, "model.mit.attn_dropout", 0.1),
                head_dim=head_dim,
                no_fusion=getattr(opts, "model.mit.no_fuse_local_global_features", False),
                conv_ksize=getattr(opts, "model.mit.conv_kernel_size", 3),
            )
        )

        return nn.Sequential(*block), input_channel



