import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
import torch.nn.functional as F


class PartialConv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        if dim > n_div:
            self.dim_conv3 = dim // n_div
        else:
            self.dim_conv3 = dim
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


class FasterMutableBlock(nn.Module):
    def __init__(self,
                 dim,
                 out_dim,
                 drop_path,
                 n_div=4,
                 mlp_ratio=2.,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 pconv_fw_type='split_cat'
                 ):

        super().__init__()

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

        self.spatial_mixing = PartialConv3(dim, n_div, pconv_fw_type)

        self.features_expand = nn.Conv2d(dim, out_dim, 1, bias=False)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))  # 直接相加而不是通道维度的concatenate
        x = self.features_expand(x)
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class SELayer(nn.Module):
    """
    Taken from:
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=[2, 3]) # Replacement of avgPool for large kernels for trt
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand(x.shape)


class PConvGuidedUpsampleBlock(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 guidance_type='full', guide_features=3):
        super(PConvGuidedUpsampleBlock, self).__init__()

        self.channel_attention = channel_attention
        self.guidance_type = guidance_type
        self.guide_features = guide_features
        self.in_features = in_features

        self.drop_path_rate = 0.1
        # 用到了7个FasterMutableBlock drop概率逐步提升
        dpr = [x.item() for x in torch.linspace(0, end=self.drop_path_rate, steps=7)]

        padding = kernel_size // 2
        self.feature_conv = nn.Sequential(
            # in_features=256, expand_features=256
            # nn.Conv2d(in_features, expand_features, kernel_size=kernel_size, padding=padding),
            FasterMutableBlock(dim=in_features, out_dim=expand_features, drop_path=dpr[0]),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            # nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            FasterMutableBlock(dim=expand_features, out_dim=expand_features // 2, drop_path=dpr[1]),
            nn.BatchNorm2d(expand_features // 2),
            nn.ReLU(inplace=True))

        if self.guidance_type == 'full':
            self.guide_conv = nn.Sequential(
                # nn.Conv2d(self.guide_features, expand_features, kernel_size=kernel_size, padding=padding),
                FasterMutableBlock(dim=self.guide_features, out_dim=expand_features, drop_path=dpr[2]),
                nn.BatchNorm2d(expand_features),
                nn.ReLU(inplace=True),
                # nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
                FasterMutableBlock(dim=expand_features, out_dim=expand_features // 2, drop_path=dpr[3]),
                nn.BatchNorm2d(expand_features // 2),
                nn.ReLU(inplace=True))

            comb_features = (expand_features // 2) * 2
        elif self.guidance_type == 'raw':
            comb_features = expand_features // 2 + guide_features
        else:
            comb_features = expand_features // 2

        self.comb_conv = nn.Sequential(
            # nn.Conv2d(comb_features, expand_features, kernel_size=kernel_size, padding=padding),
            FasterMutableBlock(dim=comb_features, out_dim=expand_features, drop_path=dpr[4]),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            # nn.Conv2d(expand_features, in_features, kernel_size=1),
            FasterMutableBlock(dim=expand_features, out_dim=in_features, drop_path=dpr[5]),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True))

        # self.reduce = nn.Conv2d(in_features, out_features, kernel_size=1)
        self.reduce = FasterMutableBlock(dim=in_features, out_dim=out_features, drop_path=dpr[6])

        if self.channel_attention:
            self.SE_block = SELayer(comb_features, reduction=1)

    def forward(self, guide, depth):
        x = self.feature_conv(depth)
        if self.guidance_type == 'full':
            y = self.guide_conv(guide)
            xy = torch.cat([x, y], dim=1)
        elif self.guidance_type == 'raw':
            xy = torch.cat([x, guide], dim=1)
        else:
            xy = x

        if self.channel_attention:
            xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        return self.reduce(residual + depth)


class BaseUpsamplingBlock(nn.Module):
    '''
    input params:
    [x]: feature
    [edge]: edge mask
    '''
    def __init__(self,
                 in_features,
                 out_features):
        super(BaseUpsamplingBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
        self.squeeze = nn.Sequential(
            nn.Conv2d(out_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0)

    def forward(self, x, edge):
        # print("x.shape:", x.shape)
        # print("edge.shape:", edge.shape)
        y = x * edge
        # print("BaseUpsamplingBlock y.shape:", y.shape)

        shortcut = y

        y = self.squeeze(self.conv1(y))
        y = shortcut + y
        y = self.conv2(y)
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')
        # print("BaseUpsamplingBlock y_.shape:", y.shape)

        return y

