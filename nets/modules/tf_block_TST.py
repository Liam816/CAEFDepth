import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, ConvModule
from typing import List


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        # nn.init.constant_(bn.weight, bn_weight_init)
        # nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class GetGlobalKV(torch.nn.Module):
#     def __init__(self, dim, key_dim, value_dim, num_heads, attn_ratio=2.0, norm_cfg=dict(type='BN', requires_grad=True)):
#         """
#         Args:
#             dim: 特征图的维度
#             key_dim: 每个head的K的维度
#             num_heads:
#             value_dim: 每个head的V维度
#             norm_cfg:
#         """
#         super().__init__()
#         self.num_heads = num_heads  # 2, 4, 5
#         self.key_dim = key_dim  # 16
#         self.value_dim = value_dim  # 32
#         self.nh_kd = nh_kd = key_dim * num_heads  # 16 * 2 = 32
#         self.d = int(attn_ratio * value_dim)  # 2.0 * 32 = 64.0
#         self.dh = int(attn_ratio * value_dim) * num_heads  # 2.0 * 32 * 2 = 128.0
#
#         # feature_channels: [64, 128, 160] -> [16*2, 16*4, 16*5] = [32, 64, 80]
#         self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
#         # feature_channels: [64, 128, 160] -> [64*2, 64*4, 64*5] = [128, 256, 320]
#         self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
#
#     def forward(self, x):
#         B, C, H, W = get_shape(x)  # [b, 64, h/32, w/32]
#
#         # [b, 64, h/32, w/32] -> [b, 32, h/32, w/32] -> [b, 2, 16, h/32 * w/32]
#         # [b, 128, h/32, w/32] -> [b, 64, h/32, w/32] -> [b, 4, 16, h/32 * w/32]
#         # [b, 160, h/32, w/32] -> [b, 80, h/32, w/32] -> [b, 5, 16, h/32 * w/32]
#         kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
#         # [b, 64, h/32, w/32] -> [b, 128, h/32, w/32] -> [b, 2, 64, h/32 * w/32] -> [b, 2, h/32 * w/32, 64]
#         # [b, 128, h/32, w/32] -> [b, 256, h/32, w/32] -> [b, 4, 64, h/32 * w/32] -> [b, 4, h/32 * w/32, 64]
#         # [b, 160, h/32, w/32] -> [b, 320, h/32, w/32] -> [b, 5, 64, h/32 * w/32] -> [b, 5, h/32 * w/32, 64]
#         vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
#
#         return kk, vv


class GetGlobalKV(torch.nn.Module):
    def __init__(self, dim, key_dim, value_dim, num_heads, attn_ratio=2.0, norm_cfg=dict(type='BN', requires_grad=True)):
        """
        Args:
            dim: 特征图的维度
            key_dim: 每个head的K的维度
            num_heads:
            value_dim: 每个head的V维度
            norm_cfg:
        """
        super().__init__()
        self.num_heads = num_heads  # 2, 4, 5
        self.key_dim = key_dim  # 16
        self.value_dim = value_dim  # 32
        self.nh_kd = nh_kd = key_dim * num_heads  # 16 * 2 = 32
        self.d = int(attn_ratio * value_dim)  # 2.0 * 32 = 64.0
        self.dh = int(attn_ratio * value_dim) * num_heads  # 2.0 * 32 * 2 = 128.0

        # feature_channels: [160, 160, 160] -> [16*2, 16*4, 16*5] = [32, 64, 80]
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        # feature_channels: [160, 160, 160] -> [64*2, 64*4, 64*5] = [128, 256, 320]
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        B, C, H, W = get_shape(x)  # [b, 64, h/32, w/32]

        # [b, 160, h/32, w/32] -> [b, 32, h/32, w/32] -> [b, 2, 16, h/32 * w/32]
        # [b, 160, h/32, w/32] -> [b, 64, h/32, w/32] -> [b, 4, 16, h/32 * w/32]
        # [b, 160, h/32, w/32] -> [b, 80, h/32, w/32] -> [b, 5, 16, h/32 * w/32]
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        # [b, 160, h/32, w/32] -> [b, 128, h/32, w/32] -> [b, 2, 64, h/32 * w/32] -> [b, 2, h/32 * w/32, 64]
        # [b, 160, h/32, w/32] -> [b, 256, h/32, w/32] -> [b, 4, 64, h/32 * w/32] -> [b, 4, h/32 * w/32, 64]
        # [b, 160, h/32, w/32] -> [b, 320, h/32, w/32] -> [b, 5, 64, h/32 * w/32] -> [b, 5, h/32 * w/32, 64]
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        return kk, vv


class GetLocalQ(torch.nn.Module):
    def __init__(self, dim, query_dim, num_heads, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads  # 2, 4, 5
        self.query_dim = query_dim  # 16
        self.nh_kd = nh_kd = query_dim * num_heads  # 32, 64, 80

        # feature_channels: [64, 128, 160] -> [16*2, 16*4, 16*5] = [32, 64, 80]
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        # [b, 64, h/32, w/32] -> [b, 32, h/32, w/32] -> [b, 2, 16, h/32 * w/32] -> [b, 2, h/32 * w/32, 16]
        # [b, 128, h/32, w/32] -> [b, 64, h/32, w/32] -> [b, 4, 16, h/32 * w/32] -> [b, 4, h/32 * w/32, 16]
        # [b, 160, h/32, w/32] -> [b, 80, h/32, w/32] -> [b, 5, 16, h/32 * w/32] -> [b, 5, h/32 * w/32, 16]
        qq = self.to_q(x).reshape(B, self.num_heads, self.query_dim, H * W).permute(0, 1, 3, 2)

        return qq


class AttentionTST(nn.Module):
    def __init__(self, local_dim, global_dim, query_dim, key_dim, value_dim, num_heads, attn_ratio=2.0, activation=nn.ReLU6,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(AttentionTST, self).__init__()
        self.dh = int(attn_ratio * value_dim) * num_heads
        self.get_kv = GetGlobalKV(global_dim, key_dim, value_dim, num_heads, attn_ratio, norm_cfg)
        self.get_q = GetLocalQ(local_dim, query_dim, num_heads, norm_cfg)
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(self.dh, local_dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, local_token, global_token):
        b, _, h, w = get_shape(local_token)
        _, _, h_, w_ = get_shape(global_token)
        assert h == h_ and w == w_, print('local_token should have the same size with global_token.')

        kk, vv = self.get_kv(global_token)
        qq = self.get_q(local_token)
        # print('qq.shape:', qq.shape)
        # print('kk.shape:', kk.shape)
        # print('vv.shape:', vv.shape)

        # [b, 2, h/32 * w/32, h/32 * w/32]
        # [b, 4, h/32 * w/32, h/32 * w/32]
        # [b, 5, h/32 * w/32, h/32 * w/32]
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = num_heads

        # [b, 2, h/32 * w/32, 64]
        # [b, 4, h/32 * w/32, 64]
        # [b, 5, h/32 * w/32, 64]
        xx = torch.matmul(attn, vv)  # [b, 8, h/64 * w/64, 32]

        # [b, 2, h/32 * w/32, 64] -> [b, 2, 64, h/32 * w/32] -> [b, 128, h/32, w/32]
        # [b, 4, h/32 * w/32, 64] -> [b, 4, 64, h/32 * w/32] -> [b, 256, h/32, w/32]
        # [b, 5, h/32 * w/32, 64] -> [b, 5, 64, h/32 * w/32] -> [b, 320, h/32, w/32]
        xx = xx.permute(0, 1, 3, 2).reshape(b, self.dh, h, w)

        # [b, 128, h/32, w/32] -> [b, 64, h/32, w/32]
        # [b, 256, h/32, w/32] -> [b, 128, h/32, w/32]
        # [b, 320, h/32, w/32] -> [b, 160, h/32, w/32]
        xx = self.proj(xx)

        return xx


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_c, hidden_c):
        super(FeedForwardNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(hidden_c, (hidden_c * 1), kernel_size=3, stride=1, padding=1, groups=hidden_c),
            nn.BatchNorm2d(hidden_c),
            nn.ReLU6(),
            nn.Conv2d(hidden_c, in_c, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)


class TFBlockTST(nn.Module):
    """
    Token-Sharing
    """
    def __init__(self, local_dim, global_dim, query_dim, key_dim, value_dim, num_heads, attn_ratio=2., mlp_ratio=4.,
                 drop=0., drop_path=0., act_layer=nn.ReLU6, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super(TFBlockTST, self).__init__()

        self.attn = AttentionTST(local_dim, global_dim, query_dim, key_dim, value_dim, num_heads, attn_ratio,
                                 activation=act_layer, norm_cfg=norm_cfg)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(local_dim * mlp_ratio)
        # self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)
        self.ffn = FeedForwardNetwork(in_c=local_dim, hidden_c=mlp_hidden_dim)

    def forward(self, local_feat, global_feat):
        # x = self.drop_path(self.attn(local_feat, global_feat))
        # x = self.drop_path(self.ffn(x))
        x = self.attn(local_feat, global_feat)
        x = self.ffn(x)

        return x


class ConnectionModule(nn.Module):
    def __init__(self, block_num, local_dims: List, global_dim, query_dims: List, key_dims: List, value_dims: List, num_heads: List,
                 attn_ratio=2., mlp_ratio=4., drop=0., attn_drop=0., drop_path_rate=0.1, act_layer=nn.ReLU6,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 ):
        super(ConnectionModule, self).__init__()
        self.block_num = block_num

        drop_path = [x.item() for x in torch.linspace(0, drop_path_rate, block_num)]
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                TFBlockTST(
                    local_dim=local_dims[i],
                    global_dim=global_dim,
                    query_dim=query_dims[i],
                    key_dim=key_dims[i],
                    value_dim=value_dims[i],
                    num_heads=num_heads[i],
                    attn_ratio=attn_ratio,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    act_layer=act_layer,
                    norm_cfg=norm_cfg
                )
            )

        self.single_conv_block = nn.Sequential(
            nn.Conv2d(local_dims[-1], local_dims[-1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(local_dims[-1]),
            nn.ReLU6(),
        )

    def forward(self, features):
        b, c, h, w = get_shape(features[-1])
        shortcut0 = features[0]  # [b, 64, h/8, w/8]
        shortcut1 = features[1]  # [b, 128, h/16, w/16]
        features[0] = nn.functional.adaptive_avg_pool2d(features[0], (h, w))
        features[1] = nn.functional.adaptive_avg_pool2d(features[1], (h, w))

        # features:
        # [b, 64, h/32, w/32]
        # [b, 128, h/32, w/32]
        # [b, 160, h/32, w/32]
        # for i in range(len(features)):
        #     print('features[{}].shape:{}'.format(i, features[i].shape))
        # exit()

        out_features = []

        x0 = self.transformer_blocks[0](features[0], features[-1])
        x0 = F.interpolate(x0, scale_factor=4.0, mode='bilinear')
        out_features.append(x0 + shortcut0)

        x1 = self.transformer_blocks[1](features[1], features[-1])
        x1 = F.interpolate(x1, scale_factor=2.0, mode='bilinear')
        out_features.append(x1 + shortcut1)

        x2 = self.transformer_blocks[2](features[-1], features[-1])
        out_features.append(x2 + features[-1])

        out_features.append(self.single_conv_block(features[-1]))

        return out_features







