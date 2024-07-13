import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn
import argparse
from typing import Dict, Tuple, Optional, Union, Any
from utils import logger

from ..layers import ConvLayer, LinearLayer, GlobalPool, Dropout, Identity
from ..modules import InvertedResidual, MobileViTBlock, Guided_Upsampling_Block
from ..modules import MobileViTBlockv2 as Block
from ..misc.init_utils import initialize_weights, initialize_fc_layer
from ..misc.common import parameter_list
from ..misc.profiler import module_profile

import sys

from utils.math_utils import make_divisible, bound_fn


def get_configuration(opts) -> Dict:

    width_multiplier = getattr(opts, "model.mitv2.width_multiplier", 1.0)

    ffn_multiplier = (
        2  # bound_fn(min_val=2.0, max_val=4.0, value=2.0 * width_multiplier)
    )
    mv2_exp_mult = 2  # max(1.0, min(2.0, 2.0 * width_multiplier))

    layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))
    config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": layer_0_dim,
        },
        "layer1": {
            "out_channels": int(make_divisible(64 * width_multiplier, divisor=16)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": int(make_divisible(128 * width_multiplier, divisor=8)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 2,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channels": int(make_divisible(256 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(128 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channels": int(make_divisible(384 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(192 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channels": int(make_divisible(512 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(256 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }

    return config


class MobileViTv2GUB(nn.Module):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ architecture
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        mobilevit_config = get_configuration(opts=opts)

        image_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]
        self.dilation = 1
        self.dilate_l4 = False
        self.dilate_l5 = False

        self.gradient_checkpointing = getattr(opts, "model.gradient_checkpointing", False)  # 默认为False

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

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

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }

        up_features = [512, 256, 128, 64, 32]
        expand_features = [512, 256, 128, 64, 32]
        # up_features = [512, 512, 512, 512, 512]
        # expand_features = [512, 512, 512, 512, 512]
        self.decoder_up1 = Guided_Upsampling_Block(in_features=up_features[0],
                                                   expand_features=expand_features[0],
                                                   out_features=up_features[1],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 256, 30, 40]

        self.decoder_up2 = Guided_Upsampling_Block(in_features=up_features[1],
                                                   expand_features=expand_features[1],
                                                   out_features=up_features[2],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")

        self.decoder_up3 = Guided_Upsampling_Block(in_features=up_features[2],
                                                   expand_features=expand_features[2],
                                                   out_features=up_features[3],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")

        self.decoder_up4 = Guided_Upsampling_Block(in_features=up_features[3],
                                                   expand_features=expand_features[3],
                                                   out_features=up_features[4],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")

        self.decoder_up5 = Guided_Upsampling_Block(in_features=up_features[4],
                                                   expand_features=expand_features[4],
                                                   out_features=1,
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")


        # # weight initialization
        # self.init_parameters(opts=opts)

    def init_parameters(self, opts):
        """Initialize model weights"""
        initialize_weights(opts=opts, modules=self.modules())
        # self.modules()是父类nn.Module中的方法

    def forward_layer(self, layer: nn.Module, x: Tensor) -> Tensor:
        # Larger models with large input image size may not be able to fit into memory.
        # We can use gradient checkpointing to enable training with large models and large inputs
        return (
            gradient_checkpoint_fn(layer, x)
            if self.gradient_checkpointing
            else layer(x)
        )

    def encoder_extract_features(self, x: Tensor, *args, **kwargs):
        features = []
        x = self.forward_layer(self.conv_1, x)  # [B, 32, 240, 320]
        features.append(x)
        x = self.forward_layer(self.layer_1, x)  # [B, 64, 240, 320]
        features.append(x)
        x = self.forward_layer(self.layer_2, x)  # [B, 128, 120, 160]
        features.append(x)
        x = self.forward_layer(self.layer_3, x)  # [B, 256, 60, 80]
        features.append(x)
        x = self.forward_layer(self.layer_4, x)  # [B, 384, 30, 40]
        features.append(x)
        x = self.forward_layer(self.layer_5, x)  # [B, 512, 16, 20]
        features.append(x)
        # x = self.forward_layer(self.conv_1x1_exp, x)  # [B, 640, 15, 20]
        # features.append(x)
        return features

    def decoder_upsample(self, x: Tensor, features, *args, **kwargs):
        y = features[-1]  # 倒数第一层特征图 [B, 512, 16, 20]
        features_ = []
        guide_maps = [x]  # [,,480, 640] [,,240, 320] [,,120, 160] [,,60, 80] [,,30, 40]
        for i in range(4):
            guide_maps.append(F.interpolate(x, scale_factor=(0.5 ** (i+1))))
        y = F.interpolate(y, size=[guide_maps[-1].size(2), guide_maps[-1].size(3)], mode='bilinear')  # [B, 512, 30, 40]
        y = self.decoder_up1(guide_maps[-1], y)  # [B, 256, 30, 40]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 256, 60, 80]
        y = self.decoder_up2(guide_maps[-2], y)  # [B, 128, 60, 80]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 128, 120, 160]
        y = self.decoder_up3(guide_maps[-3], y)  # [B, 64, 120, 160]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 64, 240, 320]
        y = self.decoder_up4(guide_maps[-4], y)  # [B, 32, 240, 320]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 32, 480, 640]
        y = self.decoder_up5(guide_maps[-5], y)  # [B, 1, 480, 640]
        features_.append(y)
        return y

    def forward(self, x: Any, *args, **kwargs) -> Any:
        # NOTE: Encoder特征提取
        features1 = self.encoder_extract_features(x, *args, **kwargs)
        # # NOTE: Decoder上采样
        features2 = self.decoder_upsample(x, features1)  # 把encoder输出的特征图列表传入
        return features2

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.classification.mitv2.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mitv2.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between FFN layers. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mitv2.dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mitv2.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier. Defaults to 1.0",
        )
        group.add_argument(
            "--model.classification.mitv2.attn-norm-layer",
            type=str,
            default="layer_norm_2d",
            help="Norm layer in attention block. Defaults to LayerNorm",
        )
        return parser

    def _make_layer(
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
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
        opts, input_channel: int, cfg: Dict
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
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
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

        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = getattr(opts, "model.classification.mitv2.dropout", 0.0)

        block.append(
            Block(
                opts=opts,
                in_channels=input_channel,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=getattr(
                    opts, "model.classification.mitv2.ffn_dropout", 0.0
                ),
                attn_dropout=getattr(
                    opts, "model.classification.mitv2.attn_dropout", 0.0
                ),
                conv_ksize=3,
                attn_norm_layer=getattr(
                    opts, "model.classification.mitv2.attn_norm_layer", "layer_norm_2d"
                ),
                dilation=self.dilation,
            )
        )

        return nn.Sequential(*block), input_channel




