import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn
from typing import Dict, Tuple, Optional, Union, Any

from .config.mobilevitv3 import get_configuration
from ..layers import ConvLayer, LinearLayer, GlobalPool, Dropout
from ..modules import InvertedResidual, Guided_Upsampling_Block, MobileViTv3Block

from typing import Dict

from utils import logger


class MobileViTv3GUB(nn.Module):
    """
        MobileViTv3:
    """
    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        image_channels = 3
        out_channels = 16

        mobilevit_config = get_configuration(opts=opts)

        self.dilation = 1
        output_stride = kwargs.get("output_stride", None)  # 默认为None
        self.dilate_l4 = False
        self.dilate_l5 = False

        self.gradient_checkpointing = getattr(opts, "model.gradient_checkpointing", False)  # 默认为False

        # store model configuration in a dictionary
        self.model_conf_dict = dict()

        ''' Encoder(MobileViTv3) '''
        self.conv_1 = ConvLayer(
                opts=opts, in_channels=image_channels, out_channels=out_channels,
                kernel_size=3, stride=2, use_norm=True, use_act=True
            )

        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict['layer1'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict['layer2'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict['layer3'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer4"], dilate=self.dilate_l4
        )
        self.model_conf_dict['layer4'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer5"], dilate=self.dilate_l5
        )
        self.model_conf_dict['layer5'] = {'in': in_channels, 'out': out_channels}

        # in_channels = out_channels
        # exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        # self.conv_1x1_exp = ConvLayer(
        #         opts=opts, in_channels=in_channels, out_channels=exp_channels,
        #         kernel_size=1, stride=1, use_act=True, use_norm=True
        #     )
        # self.model_conf_dict['exp_before_cls'] = {'in': in_channels, 'out': exp_channels}

        up_features = [320, 160, 80, 40, 20]
        expand_features = [320, 160, 80, 40, 20]
        self.decoder_up1 = Guided_Upsampling_Block(in_features=up_features[0],
                                                   expand_features=expand_features[0],
                                                   out_features=up_features[1],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 160, 30, 40]

        self.decoder_up2 = Guided_Upsampling_Block(in_features=up_features[1],
                                                   expand_features=expand_features[1],
                                                   out_features=up_features[2],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 80, 60, 80]

        self.decoder_up3 = Guided_Upsampling_Block(in_features=up_features[2],
                                                   expand_features=expand_features[2],
                                                   out_features=up_features[3],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 40, 120, 160]

        self.decoder_up4 = Guided_Upsampling_Block(in_features=up_features[3],
                                                   expand_features=expand_features[3],
                                                   out_features=up_features[4],
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 20, 240, 320]

        self.decoder_up5 = Guided_Upsampling_Block(in_features=up_features[4],
                                                   expand_features=expand_features[4],
                                                   out_features=1,
                                                   kernel_size=3,
                                                   channel_attention=True,
                                                   guide_features=3,
                                                   guidance_type="full")  # [B, 1, 480, 640]

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
        x = self.forward_layer(self.conv_1, x)  # [B, 16, 240, 320]
        features.append(x)
        x = self.forward_layer(self.layer_1, x)  # [B, 32, 240, 320]
        features.append(x)
        x = self.forward_layer(self.layer_2, x)  # [B, 64, 120, 160]
        features.append(x)
        x = self.forward_layer(self.layer_3, x)  # [B, 128, 60, 80]
        features.append(x)
        x = self.forward_layer(self.layer_4, x)  # [B, 256, 30, 40]
        features.append(x)
        x = self.forward_layer(self.layer_5, x)  # [B, 320, 15, 20]
        features.append(x)
        # x = self.forward_layer(self.conv_1x1_exp, x)  # [B, 960, 15, 20]
        # features.append(x)
        return features

    def decoder_upsample(self, x: Tensor, features, *args, **kwargs):
        y = features[-1]  # 倒数第一层特征图 [B, 320, 15, 20]
        features_ = []
        guide_maps = [x]  # [,,480, 640] [,,240, 320] [,,120, 160] [,,60, 80] [,,30, 40]
        for i in range(4):
            guide_maps.append(F.interpolate(x, scale_factor=(0.5 ** (i+1))))
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 320, 30, 40]
        y = self.decoder_up1(guide_maps[-1], y)  # [B, 160, 30, 40]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 160, 60, 80]
        y = self.decoder_up2(guide_maps[-2], y)  # [B, 80, 60, 80]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 80, 120, 160]
        y = self.decoder_up3(guide_maps[-3], y)  # [B, 40, 120, 160]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 40, 240, 320]
        y = self.decoder_up4(guide_maps[-4], y)  # [B, 20, 240, 320]
        features_.append(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear')  # [B, 20, 480, 640]
        y = self.decoder_up5(guide_maps[-5], y)  # [B, 1, 480, 640]
        features_.append(y)
        return y

    def forward(self, x: Any, *args, **kwargs) -> Any:
        # NOTE: Encoder特征提取
        features1 = self.encoder_extract_features(x, *args, **kwargs)
        # for i in range(len(features1)):
        #     print("features1[{}].shape:{}".format(i, features1[i].shape))
        # NOTE: Decoder上采样
        features2 = self.decoder_upsample(x, features1)  # 把encoder输出的特征图列表传入
        return features2

    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg,
                dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
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
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
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
                dilation=prev_dilation
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
            logger.error("Transformer input dimension should be divisible by head dimension. "
                         "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(
            MobileViTv3Block(
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
                conv_ksize=getattr(opts, "model.mit.conv_kernel_size", 3)
            )
        )

        return nn.Sequential(*block), input_channel

