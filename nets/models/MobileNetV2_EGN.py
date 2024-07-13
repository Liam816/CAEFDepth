import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn
import argparse
from typing import Dict, Tuple, Optional, List, Union, Any
from utils import logger

from ..layers import ConvLayer, LinearLayer, GlobalPool, Dropout
from ..modules import InvertedResidual, MobileViTBlock, Guided_Upsampling_Block
from ..misc.init_utils import initialize_weights, initialize_fc_layer
from ..misc.common import parameter_list
from ..misc.profiler import module_profile
from fast_transformers.builders import TransformerEncoderBuilder

import sys

from utils import logger
from utils.math_utils import make_divisible

import numpy as np
import cv2 as cv


def get_configuration(opts) -> Dict:

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


class DecoderBlock(nn.Module):
    def __init__(self, input_features, output_features):
        super(DecoderBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')
        return y


class TRFA(nn.Module):
    """
    Transformer-Based Feature Aggregation Module
    """
    def __init__(self, input_features):
        super(TRFA, self).__init__()

        builder1 = TransformerEncoderBuilder.from_kwargs(
            n_layers=1,  # 8
            n_heads=4,  # 8
            query_dimensions=75,  # 默认64 n_heads*query_dimensions=8*150=30*40=h*w
            value_dimensions=75,  # 64
            feed_forward_dimensions=128  # 1024
        )
        builder2 = TransformerEncoderBuilder.from_kwargs(
            n_layers=1,
            n_heads=4,
            query_dimensions=75,
            value_dimensions=75,
            feed_forward_dimensions=128
        )

        # Build a transformer with linear attention
        builder1.attention_type = "linear"
        builder2.attention_type = "linear"

        # LTR_encoder1 131.67M
        self.LTR_encoder1 = builder1.get()
        self.LTR_encoder2 = builder2.get()

        self.conv = nn.Sequential(
            nn.Conv2d(input_features, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # 0.03M

    def forward(self, feature, edge):
        b, c, h, w = feature.shape  # [B, 128, 30, 40]
        feature = torch.reshape(feature, (b, c, h * w))  #
        b, c, _, _ = edge.shape  # [B, 128, 30, 40]
        edge = torch.reshape(edge, (b, c, h * w))

        feature = self.LTR_encoder1(feature)  # [4, 128, 1200]
        # print("feature.shape:", feature.shape)
        edge = self.LTR_encoder2(edge)  # [4, 128, 1200]
        # print("edge.shape:", edge.shape)
        y = torch.cat((feature, edge), dim=1)
        y = torch.reshape(y, (b, c*2, h, w))
        y = self.conv(y)
        return y


class CAFF(nn.Module):
    """
    Channel Attention-Based Feature Fusion Module
    """
    def __init__(self, input_features, output_features):
        super(CAFF, self).__init__()
        # output_features = input_features // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_features),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # [b, c, 1, 1] 将每个通道的特征图求平均变成一个标量，表示通道的注意力权重
            nn.Conv2d(output_features, output_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_features, output_features, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, feature, edge):
        feature_shortcut = feature
        edge_shortcut = edge
        x = torch.cat((feature, edge), dim=1)
        x = self.conv1(x)  # attention map
        y = x * feature_shortcut + x * edge_shortcut
        return y


class EdgeHead(nn.Module):
    """
    Edge Head
    """
    def __init__(self, input_features):  # 32->1
        super(EdgeHead, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_features, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.encoder(x)
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')
        return y


class MobileNetV2EGN(nn.Module):
    """
    This class defines the `MobileNetv2 architecture <https://arxiv.org/abs/1801.04381>`_
    """
    def __init__(self, opts, *args, **kwargs) -> None:
        super(MobileNetV2EGN, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dilation = 1
        self.round_nearest = 8
        self.dilate_l4 = False

        width_mult = getattr(opts, "model.mobilenetv2.width_multiplier", 1.0)

        cfg = get_configuration(opts=opts)

        image_channels = 3
        input_channels = 32
        last_channel = 1280

        last_channel = make_divisible(last_channel * max(1.0, width_mult), self.round_nearest)

        self.model_conf_dict = dict()

        # NOTE: Backbone 0.54M
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

        # self.layer_5, out_channels = self._make_layer(
        #     opts=opts,
        #     mv2_config=[cfg["layer5"], cfg["layer5_a"]],
        #     width_mult=width_mult,
        #     input_channel=input_channels,
        #     dilate=self.dilate_l5,
        # )
        # self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
        # input_channels = out_channels
        #
        # self.conv_1x1_exp = ConvLayer(
        #     opts=opts,
        #     in_channels=input_channels,
        #     out_channels=last_channel,
        #     kernel_size=1,
        #     stride=1,
        #     use_act=True,
        #     use_norm=True,
        # )
        # self.model_conf_dict["exp_before_cls"] = {
        #     "in": input_channels,
        #     "out": last_channel,
        # }

        # NOTE: else
        self.mv2_modified_conv = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)  # <<0.01M

        self.extension_module = self._make_extension_module(opts=opts, input_channel=96)  # 0.78M

        self.edge_compact_module = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        )  # [b, 32, h/2, w/2]  0.01M

        # egb_conv 0.31M
        self.egb_conv1 = self._make_egb_conv(32, 24)  # [b, 24, h/4, w/4]
        self.egb_conv2 = self._make_egb_conv(24, 32)  # [b, 32, h/8, w/8]
        self.egb_conv3 = self._make_egb_conv(32, 96)  # [b, 96, h/16, w/16]
        self.egb_conv4 = self._make_egb_conv(96, 96, stride=1)  # [b, 96, h/16, w/16]
        self.egb_conv5 = self._make_egb_conv(96, 128, stride=1)  # [b, 128, h/16, w/16]
        self.egb_conv6 = self._make_egb_conv(248, 32, stride=1)  # [b, 32, h/4, w/4]

        # caff 0.05M
        self.caff1 = CAFF(input_features=64, output_features=32)
        self.caff2 = CAFF(input_features=48, output_features=24)
        self.caff3 = CAFF(input_features=64, output_features=32)
        self.caff4 = CAFF(input_features=192, output_features=96)

        # self.trfa = TRFA(256)  # 131.7M

        self.edge_head = EdgeHead(input_features=32)  # <<0.01M

        # decoder_block 0.20M
        self.decoder_block1 = DecoderBlock(input_features=160, output_features=96)
        self.decoder_block2 = DecoderBlock(input_features=120, output_features=64)
        self.decoder_block3 = DecoderBlock(input_features=64, output_features=1)

        # decoder_conv 0.03M
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def sobel_extractor(self, x):
        b, c, h, w = x.shape
        x_ = x
        x_ = x_ * 255
        x_ = x_.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

        sobel_batch_tensor = torch.randn(size=(b, 3, h, w))
        for i in range(b):
            Sobelx = cv.Sobel(x_[i, :, :, :], cv.CV_32F, 1, 0)
            Sobely = cv.Sobel(x_[i, :, :, :], cv.CV_32F, 0, 1)
            Sobelx = cv.convertScaleAbs(Sobelx)
            Sobely = cv.convertScaleAbs(Sobely)
            Sobelxy = cv.addWeighted(Sobelx, 0.5, Sobely, 0.5, 0).transpose(2, 0, 1)  # [c, h, w]
            Sobelxy = np.expand_dims(Sobelxy, axis=0)  # [1, 3, h, w]
            # print("Sobelxy.shape:", Sobelxy.shape)
            sobel_batch_tensor[i, :, :, :] = torch.from_numpy(Sobelxy).type(torch.float32)  # [b, c, h, w]

        return sobel_batch_tensor.to(self.device)

    def multi_scale_feature_extractor(self, x: Tensor):
        features = []
        x = self.conv_1(x)
        x = self.layer_1(x)  # [B, 16, 240, 320]
        features.append(x)
        x = self.layer_2(x)  # [B, 24, 120, 160]
        features.append(x)
        x = self.layer_3(x)  # [B, 32, 60, 80]
        features.append(x)
        x = self.layer_4(x)  # [B, 96, 30, 40]
        features.append(x)
        x = self.extension_module(x)  # [B, 128, 30, 40]
        features.append(x)
        return features

    def forward(self, x):
        # NOTE: Multi-scale Feature Extractor
        D = self.multi_scale_feature_extractor(x)
        D[0] = self.mv2_modified_conv(D[0])  # [B, 32, 240, 320]
        # for i in range(len(D)):
        #     print("D[{}].shape:{}".format(i, D[i].shape))

        # NOTE: Edge Guidance Branch
        E = self.sobel_extractor(x)  # [4, 3, 480, 640]
        # print("E.shape:{}".format(E.shape))
        E1 = self.edge_compact_module(E)  # [4, 32, 240, 320]
        # print("E1.shape:{}".format(E1.shape))
        E2 = self.egb_conv1(self.caff1(E1, D[0]))  # [4, 24, 120, 160]
        # print("E2.shape:{}".format(E2.shape))
        E3 = self.egb_conv2(self.caff2(E2, D[1]))  # [4, 32, 60, 80]
        # print("E3.shape:{}".format(E3.shape))
        E4 = self.egb_conv3(self.caff3(E3, D[2]))  # [4, 96, 30, 40]
        # print("E4.shape:{}".format(E4.shape))
        E5 = self.egb_conv4(self.caff4(E4, D[3]))  # [4, 96, 30, 40]
        # print("E5.shape:{}".format(E5.shape))
        E6 = self.egb_conv5(E5)  # [4, 128, 30, 40]
        # print("E6.shape:{}".format(E6.shape))

        E3 = F.interpolate(E3, scale_factor=2.0, mode='bilinear')  # [4, 32, 120, 160]
        E4 = F.interpolate(E4, scale_factor=4.0, mode='bilinear')  # [4, 96, 120, 160]
        E5 = F.interpolate(E5, scale_factor=4.0, mode='bilinear')  # [4, 96, 120, 160]
        Fc = self.egb_conv6(torch.cat((E2, E3, E4, E5), dim=1))  # [4, 32, 120, 160]
        Fc = F.interpolate(Fc, scale_factor=2.0, mode='bilinear')  # [4, 32, 240, 320]
        # print("Fc.shape:{}".format(Fc.shape))
        y_edge = self.edge_head(Fc)  # [4, 1, 480, 640]
        # print("y_edge.shape:{}".format(y_edge.shape))

        y = self.trfa(D[4], E6)  # [4, 128, 30, 40]
        # print("y.shape:{}".format(y.shape))
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')  # [4, 128, 60, 80]
        y = torch.cat((y, D[2]), dim=1)  # [4, 160, 60, 80]
        y = self.decoder_block1(y)  # [4, 96, 120, 160]
        # print("y.shape:{}".format(y.shape))
        y = torch.cat((y, D[1]), dim=1)  # [4, 120, 120, 160]
        y = self.decoder_block2(y)  # [4, 64, 240, 320]
        # print("y.shape:{}".format(y.shape))
        y = torch.cat((y, D[0]), dim=1)  # [4, 96, 240, 320]
        y = self.decoder_conv(y)  # [4, 32, 240, 320]
        # print("y.shape:{}".format(y.shape))
        y = torch.cat((y, Fc), dim=1)  # [4, 64, 240, 320]
        y = self.decoder_block3(y)
        # print("y.shape:{}".format(y.shape))

        return y, y_edge


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

    def _make_egb_conv(self, input_features, output_features, stride=2):
        egb_conv = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_features),
            nn.ReLU(inplace=True)
        )
        return egb_conv


