import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn
import argparse
from typing import Dict, Tuple, Optional, Union, Any
from utils import logger

from ..layers import ConvLayer, LinearLayer, GlobalPool, Dropout
from ..modules import InvertedResidual, MobileViTBlock
from ..misc.init_utils import initialize_weights, initialize_fc_layer
from ..misc.common import parameter_list

import sys

from typing import Dict

from utils import logger


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


class MobileViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    def __init__(self, opts, bottleneck_num_features=640, decoder_width=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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

        ''' Encoder(MobileViT) '''
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

        in_channels = out_channels
        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=exp_channels,
            kernel_size=1,
            stride=1,
            use_act=True,
            use_norm=True,
        )
        self.model_conf_dict["exp_before_cls"] = {"in": in_channels, "out": exp_channels}

        ''' Decoder(DenseDepth) '''
        features = int(bottleneck_num_features * decoder_width)  # 640
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.decoder_conv2 = nn.Conv2d(bottleneck_num_features, features, kernel_size=1, stride=1, padding=0)  # [bs, 640, h, w]
        self.decoder_up1 = UpSample(skip_input=features // 1 + 128, output_features=features // 2)  # 联结b4 output_channels: 320
        self.decoder_up2 = UpSample(skip_input=features // 2 + 96, output_features=features // 4)  # 联结b3 output_channels: 160
        self.decoder_up3 = UpSample(skip_input=features // 4 + 64, output_features=features // 8)  # 联结b2 output_channels: 80
        self.decoder_up4 = UpSample(skip_input=features // 8 + 32,  output_features=features // 16)  # 联结b1 output_channels: 40
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.decoder_conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)  # output_channels: 1

        # weight initialization
        # self.init_parameters(opts=opts)  # Do Nothing

        # # 可以用该方法来查看模块的各层名称和参数
        # modules = self.modules()
        # for i in modules:
        #     print("module:", i)

        # for name, param in self.named_parameters():
        #     print("name:{}".format(name))
        # res = self.named_parameters()
        # logger.log("LIAM isinstance(res, list):{}".format(isinstance(res, list)))
        # print("LIAM res.type:", type(res))

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
        x = self.forward_layer(self.conv_1, x)   # [B, 16, 240, 320]
        features.append(x)
        x = self.forward_layer(self.layer_1, x)  # [B, 32, 240, 320]
        features.append(x)
        x = self.forward_layer(self.layer_2, x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.forward_layer(self.layer_3, x)  # [B, 96, 60, 80]
        features.append(x)
        x = self.forward_layer(self.layer_4, x)  # [B, 128, 30, 40]
        features.append(x)
        x = self.forward_layer(self.layer_5, x)  # [B, 160, 15, 20]
        features.append(x)
        x = self.forward_layer(self.conv_1x1_exp, x)  # [B, 640, 15, 20]
        features.append(x)
        bottleneck_num_features = x.shape[1]
        return features

    def decoder_upsample(self, x, *args, **kwargs):
        features = []
        x_block0, x_block1, x_block2, x_block3, x_block4, x_block5, x_block6 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        x_d0 = self.decoder_conv2(F.relu(x_block6))  # input: [B, 640, 15, 20],      output: [B, 640, 15, 20]
        features.append(x_d0)
        x_d1 = self.decoder_up1(x_d0, x_block4)      # input: [B, 640+128, 15, 20],  output: [bs, 320, 30, 40]
        features.append(x_d1)
        x_d2 = self.decoder_up2(x_d1, x_block3)      # input: [bs, 320+96, 30, 40],  output: [bs, 160, 60, 80]
        features.append(x_d2)
        x_d3 = self.decoder_up3(x_d2, x_block2)      # input: [bs, 160+64, 60, 80],  output: [bs, 80, 120, 160]
        features.append(x_d3)
        x_d4 = self.decoder_up4(x_d3, x_block1)      # input: [bs, 80+32, 120, 160], output: [bs, 40, 240, 320]
        features.append(x_d4)
        x_d5 = self.decoder_conv3(x_d4)              # output: [bs, 1, 240, 320]
        features.append(x_d5)
        return features

    def forward(self, x: Any, *args, **kwargs) -> Any:
        # NOTE: Encoder特征提取
        features1 = self.encoder_extract_features(x, *args, **kwargs)
        # NOTE: Decoder上采样
        features2 = self.decoder_upsample(features1)  # 把encoder输出的特征图列表传入
        return features2[-1]

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

