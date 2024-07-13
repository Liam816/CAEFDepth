import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, ConvModule
import numpy as np
import matplotlib.pyplot as plt

# print('import succeed.')
# exit()


def normalize2img_tensor(x):
    min_val = x.min()
    max_val = x.max()
    res = (x - min_val) / (max_val - min_val)
    res = res * 255.
    return res


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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


# NOTE: 原版
class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4.0,
                 activation=None,  # nn.ReLU6
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads  # 8
        self.scale = key_dim ** -0.5  # 16的-0.5次方 等于 1/4=0.25
        self.key_dim = key_dim  # 16
        self.nh_kd = nh_kd = key_dim * num_heads  # 16 * 8 = 128
        self.d = int(attn_ratio * key_dim)  # 2.0 * 16 = 32.0
        self.dh = int(attn_ratio * key_dim) * num_heads  # 32.0 * 8 = 256.0
        self.attn_ratio = attn_ratio  # 2

        # dim = 384 也即patch_embedding的维度（特征数）
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 256

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x):  # x (B,N,C)
        # x shape: [b, 224, 480/16, 640/16] = [b, 224, 30, 40]
        B, C, H, W = get_shape(x)
        # print('x.shape:', x.shape)

        # to_q: [b, 128, 30, 40]  reshape: [b, 8, 16, 30 * 40]  permute: [b, 8, 30 * 40, 16]
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        # to_k: [b, 128, 30, 40]  reshape: [b, 8, 16, 30, 40]
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        # to_v: [b, 256, 30, 40]  reshape: [b, 8, 32, 30, 40]  permute: [b, 8, 15, 20, 32]
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        # print('qq.shape:', qq.shape)
        # print('kk.shape:', kk.shape)
        # print('vv.shape:', vv.shape)

        attn = torch.matmul(qq, kk)  # [b, 8, h/64 * w/64, h/64 * w/64]
        # temp = attn  # 用以查看qk计算的相似性的直方图分布
        attn = attn.softmax(dim=-1)  # dim = k
        # temp = attn

        xx = torch.matmul(attn, vv)  # [b, 8, h/64 * w/64, 32]
        # print('xx max:{} min:{}'.format(torch.max(xx), torch.min(xx)))

        # permute: [b, 8, 32, h/64 * w/64]  reshape: [b, 256, h/64, w/64]
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        # print('xx max:{} min:{}'.format(torch.max(xx), torch.min(xx)))

        xx = self.proj(xx)  # [b, 256, h/64, w/64] -> [b, 384, h/64, w/64]
        # print('xx max:{} min:{}'.format(torch.max(xx), torch.min(xx)))

        return xx  # xx, temp


# NOTE: 可视化注意力图
# class Attention(torch.nn.Module):
#     def __init__(self, dim, key_dim, num_heads,
#                  attn_ratio=4.0,
#                  activation=None,  # nn.ReLU6
#                  norm_cfg=dict(type='BN', requires_grad=True), ):
#         super().__init__()
#         self.num_heads = num_heads  # 8
#         self.scale = key_dim ** -0.5  # 16的-0.5次方 等于 1/4=0.25
#         self.key_dim = key_dim  # 16
#         self.nh_kd = nh_kd = key_dim * num_heads  # 16 * 8 = 128
#         self.d = int(attn_ratio * key_dim)  # 2.0 * 16 = 32.0
#         self.dh = int(attn_ratio * key_dim) * num_heads  # 32.0 * 8 = 256.0
#         self.attn_ratio = attn_ratio  # 2
#
#         # dim = 384 也即patch_embedding的维度（特征数）
#         self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128
#         self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128
#         self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 256
#
#         self.proj = torch.nn.Sequential(activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
#
#     def forward(self, x):  # x (B,N,C)
#         # x shape: [b, 224, 480/16, 640/16] = [b, 224, 30, 40]
#         B, C, H, W = get_shape(x)
#         # print('x.shape:', x.shape)
#
#         # to_q: [b, 128, 30, 40]  reshape: [b, 8, 16, 30 * 40]  permute: [b, 8, 30 * 40, 16]
#         qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
#         # to_k: [b, 128, 30, 40]  reshape: [b, 8, 16, 30, 40]
#         kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
#         # to_v: [b, 256, 30, 40]  reshape: [b, 8, 32, 30, 40]  permute: [b, 8, 15, 20, 32]
#         vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
#
#         # print('qq.shape:', qq.shape)
#         # print('kk.shape:', kk.shape)
#         # print('vv.shape:', vv.shape)
#
#         attn = torch.matmul(qq, kk)  # [b, 8, h/64 * w/64, h/64 * w/64]
#         # attn_score = attn  # 用以查看qk计算的相似性的直方图分布
#         attn = attn.softmax(dim=-1)  # dim = k
#         attn_score = attn
#
#         xx = torch.matmul(attn, vv)  # [b, 8, h/64 * w/64, 32]
#         # print('xx max:{} min:{}'.format(torch.max(xx), torch.min(xx)))
#
#         # permute: [b, 8, 32, h/64 * w/64]  reshape: [b, 256, h/64, w/64]
#         xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
#         # print('xx max:{} min:{}'.format(torch.max(xx), torch.min(xx)))
#
#         xx = self.proj(xx)  # [b, 256, h/64, w/64] -> [b, 384, h/64, w/64]
#         # print('xx max:{} min:{}'.format(torch.max(xx), torch.min(xx)))
#
#         return xx, attn_score  # xx, attn_score


class GetRGBQKV(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4.0,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads  # 8
        self.key_dim = key_dim  # 16
        self.nh_kd = nh_kd = key_dim * num_heads  # 16 * 8 = 128
        self.d = int(attn_ratio * key_dim)  # 2.0 * 16 = 32.0
        self.dh = int(attn_ratio * key_dim) * num_heads  # 32.0 * 8 = 256.0
        self.attn_ratio = attn_ratio  # 2

        # dim = 384 也即patch_embedding的维度（特征数）
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 256

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        res1 = qq
        res2 = vv

        attn = torch.matmul(qq, kk)  # [b, 8, h/64 * w/64, h/64 * w/64]
        # res3 = attn  # 用以查看qk计算的相似性的直方图分布
        attn = attn.softmax(dim=-1)  # dim = k
        res3 = attn  # 因为rgb和edge的值域domain不同 所以要返回的是softmax后的qk相似性分数

        return res1, res2, res3


class GetEdgeK(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads  # 8
        self.key_dim = key_dim  # 16
        self.nh_kd = nh_kd = key_dim * num_heads  # 16 * 8 = 128

        # dim = 384 也即patch_embedding的维度（特征数）
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)

        return kk


class GetRGBQ(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads  # 8
        self.key_dim = key_dim  # 16
        self.nh_kd = nh_kd = key_dim * num_heads  # 16 * 8 = 128

        # dim = 384 也即patch_embedding的维度（特征数）
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)

        return qq


class GetRGBKV(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4.0,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads  # 8
        self.key_dim = key_dim  # 16
        self.nh_kd = nh_kd = key_dim * num_heads  # 16 * 8 = 128
        self.d = int(attn_ratio * key_dim)  # 2.0 * 16 = 32.0
        self.dh = int(attn_ratio * key_dim) * num_heads  # 32.0 * 8 = 256.0
        self.attn_ratio = attn_ratio  # 2

        # dim = 384 也即patch_embedding的维度（特征数）
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 256

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        return kk, vv


class CrossModalAttention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,  # nn.ReLU6
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads  # 8
        self.scale = key_dim ** -0.5  # 16的-0.5次方 等于 1/4=0.25
        self.key_dim = key_dim  # 16
        self.nh_kd = nh_kd = key_dim * num_heads  # 16 * 8 = 128
        self.d = int(attn_ratio * key_dim)  # 2.0 * 16 = 32.0
        self.dh = int(attn_ratio * key_dim) * num_heads  # 32.0 * 8 = 256.0
        self.attn_ratio = attn_ratio  # 2

        # dim = 384 也即patch_embedding的维度（特征数）
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 128
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)  # feature_channels: 384 -> 256

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, rgb, edge):  # x (B,N,C)
        # x shape: [b, 384, 480/64, 640/64] = [b, 384, 8, 10]
        B, C, H, W = get_shape(rgb)
        _, c, h, w = get_shape(edge)

        if C != c or H != h or W != w:
            print('RGB embedding should have the same shape with edge embedding.')
            exit()

        # print('x.shape:', x.shape)

        # to_q: [b, 128, 30, 40]  reshape: [b, 8, 16, 30 * 40]  permute: [b, 8, 30 * 40, 16]
        qq = self.to_q(rgb).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        # to_k: [b, 128, 30, 40]  reshape: [b, 8, 16, 30, 40]
        kk = self.to_k(edge).reshape(B, self.num_heads, self.key_dim, H * W)
        # to_v: [b, 256, 30, 40]  reshape: [b, 8, 32, 30 * 40]  permute: [b, 8, 30 * 40, 32]
        vv = self.to_v(edge).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        # print('qq.shape:', qq.shape)
        # print('kk.shape:', kk.shape)
        # print('vv.shape:', vv.shape)

        attn = torch.matmul(qq, kk)  # [b, 8, h/64 * w/64, h/64 * w/64]
        # temp = attn  # 用以查看qk计算的相似性的直方图分布
        attn = attn.softmax(dim=-1)  # dim = k
        # temp = attn

        xx = torch.matmul(attn, vv)  # [b, 8, h/64 * w/64, 32]

        # permute: [b, 8, 32, h/64 * w/64]  reshape: [b, 256, h/64, w/64]
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)  # [b, 256, h/64, w/64] -> [b, 384, h/64, w/64]

        return xx  # xx, temp


# NOTE: 原版
class TFBlock(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer, norm_cfg=norm_cfg)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


# NOTE: 可视化注意力图
# class TFBlock(nn.Module):
#
#     def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
#                  drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#
#         self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer, norm_cfg=norm_cfg)
#
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)
#
#     def forward(self, x1):
#         temp, attn_score = self.attn(x1)
#         x1 = x1 + self.drop_path(temp)
#         x1 = x1 + self.drop_path(self.mlp(x1))
#         return x1, attn_score


class CrossModalTFBlock(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True), alpha=0.5):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        self.alpha = alpha
        print('weighted sum alpha: {}'.format(alpha))
        self.ppa = PyramidPoolAgg(stride=2)

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.rgb_attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                  activation=act_layer, norm_cfg=norm_cfg)
        self.cm_attn = CrossModalAttention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                           activation=act_layer, norm_cfg=norm_cfg)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, rgb_feats, edge_feats):
        # x1 = x1 + self.drop_path(self.attn(x1))
        # x1 = x1 + self.drop_path(self.mlp(x1))

        rgb_feat = self.ppa(rgb_feats)  # RGB特征图
        shortcut_rgb_feat = rgb_feat
        edge_feat = self.ppa(edge_feats)  # edge特征图

        # rgb_attn, temp1 = self.rgb_attn(rgb_feat)  # RGB经过MHSA后的特征图
        # cm_attn, temp2 = self.cm_attn(rgb_feat, edge_feat)

        # # print('rgb_attn:\n', rgb_attn)
        # # exit()
        # print('rgb_attn.shape:{}'.format(rgb_attn.shape))
        # print('cm_attn.shape:{}'.format(cm_attn.shape))
        # print('rgb_attn max:{} min:{}'.format(torch.max(rgb_attn), torch.min(rgb_attn)))
        # print('cm_attn max:{} min:{}'.format(torch.max(cm_attn), torch.min(cm_attn)))
        # # exit()
        # print('temp1.shape:{}'.format(temp1.shape))  # [b, head_nums, patch_nums, patch_nums]
        # print('temp2.shape:{}'.format(temp2.shape))  # [b, head_nums, patch_nums, patch_nums]
        # print('temp1 max:{} min:{}'.format(torch.max(temp1), torch.min(temp1)))
        # print('temp2 max:{} min:{}'.format(torch.max(temp2), torch.min(temp2)))
        #
        # # rgb_attn = torch.mean(rgb_attn, dim=1)
        # # cm_attn = torch.mean(cm_attn, dim=1)
        # # print('rgb_attn.shape:{}'.format(rgb_attn.shape))
        # # print('cm_attn.shape:{}'.format(cm_attn.shape))
        # # print('rgb_attn max:{} min:{}'.format(torch.max(rgb_attn), torch.min(rgb_attn)))
        # # print('cm_attn max:{} min:{}'.format(torch.max(cm_attn), torch.min(cm_attn)))
        #
        # # x1 = normalize2img_tensor(temp1[0][0]).detach().cpu().numpy().astype(np.uint8)
        # # x2 = normalize2img_tensor(temp2[0][0]).detach().cpu().numpy().astype(np.uint8)
        # # x1 = temp1[0][0].detach().cpu().numpy().astype(np.float32)
        # # x2 = temp2[0][0].detach().cpu().numpy().astype(np.float32)
        # x1 = normalize2img_tensor(rgb_attn[0][0]).detach().cpu().numpy().astype(np.uint8)
        # x2 = normalize2img_tensor(cm_attn[0][0]).detach().cpu().numpy().astype(np.uint8)
        # print('x1 max:{} min:{}'.format(np.max(x1), np.min(x1)))
        # print('x2 max:{} min:{}'.format(np.max(x2), np.min(x2)))
        #
        # plt.figure(figsize=(7, 6))
        # plt.subplot(221), plt.imshow(x1)  # 原图
        # plt.subplot(222), plt.hist(x1.ravel(), 256)
        # plt.subplot(223), plt.imshow(x2)  # 原图
        # plt.subplot(224), plt.hist(x2.ravel(), 256)
        # plt.show()
        # exit()

        rgb_attn = self.rgb_attn(rgb_feat)  # RGB经过MHSA后的特征图
        cm_attn = self.cm_attn(rgb_feat, edge_feat)
        # print('rgb_attn max:{} min:{}'.format(torch.max(rgb_attn), torch.min(rgb_attn)))
        # print('cm_attn max:{} min:{}'.format(torch.max(cm_attn), torch.min(cm_attn)))
        cm_attn = rgb_attn + self.alpha * cm_attn  # 注意力强加的特征图加权相加
        # print('rgb_attn after addition max:{} min:{}'.format(torch.max(rgb_attn), torch.min(rgb_attn)))
        # print('shortcut_rgb_feat max:{} min:{}'.format(torch.max(shortcut_rgb_feat), torch.min(shortcut_rgb_feat)))
        x = shortcut_rgb_feat + self.drop_path(cm_attn)
        x = x + self.drop_path(self.mlp(x))

        return x


class CrossModalTFBlockV1M1(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True), alpha=0.5):
        super().__init__()
        print('class name: {}'.format(self.__class__))

        self.alpha = alpha
        print('weighted sum alpha: {}'.format(alpha))
        self.ppa = PyramidPoolAgg(stride=2)

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.rgb_attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                  activation=act_layer, norm_cfg=norm_cfg)
        self.cm_attn = CrossModalAttention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                           activation=act_layer, norm_cfg=norm_cfg)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, rgb_feats, edge_feats):
        # x1 = x1 + self.drop_path(self.attn(x1))
        # x1 = x1 + self.drop_path(self.mlp(x1))

        rgb_feat = self.ppa(rgb_feats)  # RGB特征图
        shortcut_rgb_feat = rgb_feat
        edge_feat = self.ppa(edge_feats)  # edge特征图
        shortcut_edge_feat = edge_feat

        rgb_attn = self.rgb_attn(rgb_feat)  # RGB经过MHSA后的特征图
        cm_attn = self.cm_attn(rgb_feat, edge_feat)
        # print('rgb_attn max:{} min:{}'.format(torch.max(rgb_attn), torch.min(rgb_attn)))
        # print('cm_attn max:{} min:{}'.format(torch.max(cm_attn), torch.min(cm_attn)))
        cm_attn = rgb_attn + self.alpha * cm_attn  # 注意力强加的特征图加权相加
        # print('rgb_attn after addition max:{} min:{}'.format(torch.max(rgb_attn), torch.min(rgb_attn)))
        # print('shortcut_rgb_feat max:{} min:{}'.format(torch.max(shortcut_rgb_feat), torch.min(shortcut_rgb_feat)))
        x = shortcut_rgb_feat + shortcut_edge_feat + self.drop_path(cm_attn)
        x = x + self.drop_path(self.mlp(x))

        return x


class CrossModalTFBlockV2(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True), alpha=0.5):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        self.alpha = alpha
        print('weighted sum alpha: {}'.format(alpha))
        self.ppa = PyramidPoolAgg(stride=2)

        self.dim = dim
        self.num_heads = num_heads
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.mlp_ratio = mlp_ratio

        self.rgb_qkv = GetRGBQKV(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, norm_cfg=norm_cfg)
        self.edge_k = GetEdgeK(dim, key_dim=key_dim, num_heads=num_heads, norm_cfg=norm_cfg)
        self.proj = torch.nn.Sequential(act_layer(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, rgb_feats, edge_feats):
        rgb_feat = self.ppa(rgb_feats)  # RGB特征图
        shortcut_rgb_feat = rgb_feat
        edge_feat = self.ppa(edge_feats)  # edge特征图

        B, C, H, W = get_shape(rgb_feat)
        _, c, h, w = get_shape(edge_feat)
        if C != c or H != h or W != w:
            print('RGB embedding should have the same shape with edge embedding.')
            exit()

        rgb_q, rgb_v, rgb_attn = self.rgb_qkv(rgb_feat)
        edge_k = self.edge_k(edge_feat)
        # print('rgb_q shape: {}'.format(rgb_q.shape))
        # print('rgb_v shape: {}'.format(rgb_v.shape))
        # print('rgb_attn shape: {}'.format(rgb_attn.shape))
        # print('edge_k shape: {}'.format(edge_k.shape))

        cm_attn = torch.matmul(rgb_q, edge_k)  # cross-modal attention score
        cm_attn = cm_attn.softmax(dim=-1)  # dim = k

        cm_attn = rgb_attn + self.alpha * cm_attn

        xx = torch.matmul(cm_attn, rgb_v)  # [b, 8, h/64 * w/64, 32]
        # print('xx max:{} min:{}'.format(torch.max(xx), torch.min(xx)))

        # permute: [b, 8, 32, h/64 * w/64]  reshape: [b, 256, h/64, w/64]
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        # print('xx max:{} min:{}'.format(torch.max(xx), torch.min(xx)))

        xx = self.proj(xx)  # [b, 256, h/64, w/64] -> [b, 384, h/64, w/64]

        x = shortcut_rgb_feat + self.drop_path(xx)
        x = x + self.drop_path(self.mlp(x))

        return x


# class PyramidPoolAgg(nn.Module):
#     def __init__(self, stride):
#         super().__init__()
#         self.stride = stride
#
#     def forward(self, inputs):
#         B, C, H, W = get_shape(inputs[-1])  # encoder最后一层的输出形状[b, c, h/32, w/32]
#         H = (H - 1) // self.stride + 1  # 进一步下采样h/64
#         W = (W - 1) // self.stride + 1  # 进一步下采样w/64
#         # 输出的形状仍然是[b, c, h/64, w/64] encoder四层的输出在通道维度上cat 分辨率都是原始的1/64
#         return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

class CrossModalTFBlockV3(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True), alpha=0.5, fm_res=32):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        self.alpha = alpha
        print('weighted sum alpha: {}'.format(alpha))
        self.fm_res = fm_res
        self.pool = None  # 平均池化

        self.dim = dim
        self.num_heads = num_heads
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.mlp_ratio = mlp_ratio

        print('dim:{} key_dim:{} self.dh:{}'.format(dim, key_dim, self.dh))

        self.rgb_qkv = GetRGBQKV(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, norm_cfg=norm_cfg)
        self.edge_k = GetEdgeK(dim, key_dim=key_dim, num_heads=num_heads, norm_cfg=norm_cfg)
        self.proj = torch.nn.Sequential(act_layer(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, rgb_feat, edge_feat):
        B, C, H, W = get_shape(rgb_feat)
        _, c, h, w = get_shape(edge_feat)
        if C != c or H != h or W != w:
            print('RGB embedding should have the same shape with edge embedding.')
            exit()

        # print('rgb_feat.shape:{}'.format(rgb_feat.shape))
        # print('edge_feat.shape:{}'.format(edge_feat.shape))

        shortcut_rgb_feat = rgb_feat
        shortcut_edge_feat = edge_feat

        H_ = (H - 1) // (64 // self.fm_res) + 1  # 进一步下采样到h/64
        W_ = (W - 1) // (64 // self.fm_res) + 1  # 进一步下采样到w/64

        # print('H_:{} W_:{}'.format(H_, W_))

        rgb_token = nn.functional.adaptive_avg_pool2d(rgb_feat, (H_, W_))  # RGB特征图
        shortcut_rgb_token = rgb_token
        edge_token = nn.functional.adaptive_avg_pool2d(edge_feat, (H_, W_))  # edge特征图

        # print('rgb_token.shape:{}'.format(rgb_token.shape))
        # print('edge_token.shape:{}'.format(edge_token.shape))

        rgb_q, rgb_v, rgb_attn = self.rgb_qkv(rgb_token)
        edge_k = self.edge_k(edge_token)
        # print('rgb_q shape: {}'.format(rgb_q.shape))
        # print('rgb_v shape: {}'.format(rgb_v.shape))
        # print('rgb_attn shape: {}'.format(rgb_attn.shape))
        # print('edge_k shape: {}'.format(edge_k.shape))

        cm_attn = torch.matmul(rgb_q, edge_k)  # cross-modal attention score
        cm_attn = cm_attn.softmax(dim=-1)  # dim = k

        cm_attn = rgb_attn + self.alpha * cm_attn

        xx = torch.matmul(cm_attn, rgb_v)  # [b, 8, h/64 * w/64, 32]
        # permute: [b, 8, 32, h/64 * w/64]  reshape: [b, 256, h/64, w/64]
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H_, W_)
        xx = self.proj(xx)  # [b, 256, h/64, w/64] -> [b, 384, h/64, w/64]

        x = shortcut_rgb_token + self.drop_path(xx)
        x = x + self.drop_path(self.mlp(x))
        # print('x.shape:{}'.format(x.shape))

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        # print('x.shape:{}'.format(x.shape))
        x = x + shortcut_rgb_feat + shortcut_edge_feat
        # print('x.shape:{}'.format(x.shape))

        return x


class CrossModalTFBlockV3M1(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True), alpha=0.5, fm_res=32):
        super().__init__()
        # print('class name: {}'.format(self.__class__))
        self.alpha = alpha
        # print('weighted sum alpha: {}'.format(alpha))
        self.fm_res = fm_res
        self.pool = None  # 平均池化

        self.dim = dim
        self.num_heads = num_heads
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.mlp_ratio = mlp_ratio

        # print('dim:{} key_dim:{} self.dh:{}'.format(dim, key_dim, self.dh))

        self.rgb_qkv = GetRGBQKV(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, norm_cfg=norm_cfg)
        self.edge_k = GetEdgeK(dim, key_dim=key_dim, num_heads=num_heads, norm_cfg=norm_cfg)
        self.proj = torch.nn.Sequential(act_layer(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, rgb_feat, edge_feat):
        B, C, H, W = get_shape(rgb_feat)
        _, c, h, w = get_shape(edge_feat)
        if C != c or H != h or W != w:
            print('RGB embedding should have the same shape with edge embedding.')
            exit()

        # print('rgb_feat.shape:{}'.format(rgb_feat.shape))
        # print('edge_feat.shape:{}'.format(edge_feat.shape))

        shortcut_rgb_feat = rgb_feat
        shortcut_edge_feat = edge_feat

        H_ = (H - 1) // (64 // self.fm_res) + 1  # 进一步下采样到h/64
        W_ = (W - 1) // (64 // self.fm_res) + 1  # 进一步下采样到w/64

        # print('H_:{} W_:{}'.format(H_, W_))

        rgb_token = nn.functional.adaptive_avg_pool2d(rgb_feat, (H_, W_))  # RGB特征图
        shortcut_rgb_token = rgb_token
        edge_token = nn.functional.adaptive_avg_pool2d(edge_feat, (H_, W_))  # edge特征图

        # print('rgb_token.shape:{}'.format(rgb_token.shape))
        # print('edge_token.shape:{}'.format(edge_token.shape))

        rgb_q, rgb_v, rgb_attn = self.rgb_qkv(rgb_token)
        edge_k = self.edge_k(edge_token)
        # print('rgb_q shape: {}'.format(rgb_q.shape))
        # print('rgb_v shape: {}'.format(rgb_v.shape))
        # print('rgb_attn shape: {}'.format(rgb_attn.shape))
        # print('edge_k shape: {}'.format(edge_k.shape))

        cm_attn = torch.matmul(rgb_q, edge_k)  # cross-modal attention score
        cm_attn = cm_attn.softmax(dim=-1)  # dim = k

        cm_attn = rgb_attn + self.alpha * cm_attn

        xx = torch.matmul(cm_attn, rgb_v)  # [b, 8, h/64 * w/64, 32]
        # permute: [b, 8, 32, h/64 * w/64]  reshape: [b, 256, h/64, w/64]
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H_, W_)
        xx = self.proj(xx)  # [b, 256, h/64, w/64] -> [b, 384, h/64, w/64]

        x = shortcut_rgb_token + self.drop_path(xx)
        x = x + self.drop_path(self.mlp(x))
        # print('x.shape in CrossModalTFBlockV3M1:{}'.format(x.shape))

        return x


class CrossModalTFBlockV3M2(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True), alpha=0.5, fm_res=32,
                 sn=2, global_channels=160):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        self.alpha = alpha
        print('weighted sum alpha: {}'.format(alpha))
        self.fm_res = fm_res
        self.sn = sn
        print('serial number: {}'.format(sn))
        self.pool = None  # 平均池化

        self.dim = dim
        self.num_heads = num_heads
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.mlp_ratio = mlp_ratio
        print('dim:{} key_dim:{} self.dh:{}'.format(dim, key_dim, self.dh))

        if sn != 1:
            self.rgb_l_q = GetRGBQ(dim, key_dim=key_dim, num_heads=num_heads, norm_cfg=norm_cfg)
            self.rgb_g_kv = GetRGBKV(global_channels, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, norm_cfg=norm_cfg)
            # self.fuse_conv = nn.Conv2d(dim + global_channels, dim, kernel_size=1, stride=1, padding=0)
        else:
            self.rgb_qkv = GetRGBQKV(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, norm_cfg=norm_cfg)

        self.edge_k = GetEdgeK(dim, key_dim=key_dim, num_heads=num_heads, norm_cfg=norm_cfg)
        self.proj = torch.nn.Sequential(act_layer(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, rgb_feat, edge_feat, rgb_g_feat=None):
        B, C, H, W = get_shape(rgb_feat)
        _, c, h, w = get_shape(edge_feat)
        if C != c or H != h or W != w:
            print('RGB embedding should have the same shape with edge embedding.')
            exit()
        # print('rgb_feat.shape:{}'.format(rgb_feat.shape))
        # print('edge_feat.shape:{}'.format(edge_feat.shape))
        H_ = (H - 1) // (64 // self.fm_res) + 1  # 进一步下采样到h/64
        W_ = (W - 1) // (64 // self.fm_res) + 1  # 进一步下采样到w/64

        if self.sn != 1:
            shortcut_rgb_l_feat = rgb_feat
            # shortcut_rgb_g_feat = rgb_g_feat
            shortcut_edge_feat = edge_feat

            rgb_l_token = nn.functional.adaptive_avg_pool2d(rgb_feat, (H_, W_))  # RGB特征图
            rgb_g_token = nn.functional.adaptive_avg_pool2d(rgb_g_feat, (H_, W_))  # RGB特征图
            shortcut_rgb_l_token = rgb_l_token
            edge_token = nn.functional.adaptive_avg_pool2d(edge_feat, (H_, W_))  # edge特征图

            rgb_l_q = self.rgb_l_q(rgb_l_token)
            rgb_g_k, rgb_g_v = self.rgb_g_kv(rgb_g_token)
            edge_k = self.edge_k(edge_token)
            # print('rgb_l_q.shape:{}'.format(rgb_l_q.shape))
            # print('rgb_g_k.shape:{}'.format(rgb_g_k.shape))
            # print('rgb_g_v.shape:{}'.format(rgb_g_v.shape))
            # print('edge_k.shape:{}'.format(edge_k.shape))

            rgb_attn = torch.matmul(rgb_l_q, rgb_g_k)
            rgb_attn = rgb_attn.softmax(dim=-1)

            cm_attn = torch.matmul(rgb_l_q, edge_k)
            cm_attn = cm_attn.softmax(dim=-1)

            cm_attn = rgb_attn + self.alpha * cm_attn

            xx = torch.matmul(cm_attn, rgb_g_v)  # [b, 8, h/64 * w/64, 32]
            xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H_, W_)
            xx = self.proj(xx)  # [b, 256, h/64, w/64] -> [b, 384, h/64, w/64]

            x = shortcut_rgb_l_token + self.drop_path(xx)
            x = x + self.drop_path(self.mlp(x))

            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            # shortcut_rgb_feat = self.fuse_conv(torch.cat((shortcut_rgb_l_feat, shortcut_rgb_g_feat), dim=1))
            # x = x + shortcut_rgb_feat + shortcut_edge_feat
            x = x + shortcut_rgb_l_feat + shortcut_edge_feat

        else:
            shortcut_rgb_feat = rgb_feat
            shortcut_edge_feat = edge_feat

            rgb_token = nn.functional.adaptive_avg_pool2d(rgb_feat, (H_, W_))  # RGB特征图
            shortcut_rgb_token = rgb_token
            edge_token = nn.functional.adaptive_avg_pool2d(edge_feat, (H_, W_))  # edge特征图

            rgb_q, rgb_v, rgb_attn = self.rgb_qkv(rgb_token)
            edge_k = self.edge_k(edge_token)

            cm_attn = torch.matmul(rgb_q, edge_k)
            cm_attn = cm_attn.softmax(dim=-1)

            cm_attn = rgb_attn + self.alpha * cm_attn

            xx = torch.matmul(cm_attn, rgb_v)  # [b, 8, h/64 * w/64, 32]
            xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H_, W_)
            xx = self.proj(xx)  # [b, 256, h/64, w/64] -> [b, 384, h/64, w/64]

            x = shortcut_rgb_token + self.drop_path(xx)
            x = x + self.drop_path(self.mlp(x))

            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x + shortcut_rgb_feat + shortcut_edge_feat

        return x


class RelayCrossModalTFBlockV1(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True), alpha=0.5, fm_res=32):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        self.alpha = alpha
        print('weighted sum alpha: {}'.format(alpha))
        self.fm_res = fm_res
        self.pool = None  # 平均池化

        self.dim = dim
        self.num_heads = num_heads
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.mlp_ratio = mlp_ratio

        # print('dim:{} key_dim:{} self.dh:{}'.format(dim, key_dim, self.dh))

        self.rgb_qkv = GetRGBQKV(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, norm_cfg=norm_cfg)
        self.edge_k = GetEdgeK(dim, key_dim=key_dim, num_heads=num_heads, norm_cfg=norm_cfg)
        self.proj = torch.nn.Sequential(act_layer(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)

    def forward(self, rgb_token, edge_feat):
        # print('rgb_token.shape:{}'.format(rgb_token.shape))
        # print('edge_feat.shape:{}'.format(edge_feat.shape))

        shortcut_rgb_token = rgb_token
        # rgb_token已经是1/64尺寸的  [b, c, h/64, w/64]
        B, C, H, W = get_shape(edge_feat)
        H_ = (H - 1) // (64 // self.fm_res) + 1  # 进一步下采样到h/64
        W_ = (W - 1) // (64 // self.fm_res) + 1  # 进一步下采样到w/64

        edge_token = nn.functional.adaptive_avg_pool2d(edge_feat, (H_, W_))  # edge特征图
        # print('edge_token.shape:{}'.format(edge_token.shape))
        # exit()

        rgb_q, rgb_v, rgb_attn = self.rgb_qkv(rgb_token)
        edge_k = self.edge_k(edge_token)

        cm_attn = torch.matmul(rgb_q, edge_k)  # cross-modal attention score
        cm_attn = cm_attn.softmax(dim=-1)  # dim = k

        cm_attn = rgb_attn + self.alpha * cm_attn

        xx = torch.matmul(cm_attn, rgb_v)  # [b, 8, h/64 * w/64, 32]
        # permute: [b, 8, 32, h/64 * w/64]  reshape: [b, 256, h/64, w/64]
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H_, W_)
        xx = self.proj(xx)  # [b, 256, h/64, w/64] -> [b, 384, h/64, w/64]

        x = shortcut_rgb_token + self.drop_path(xx)
        x = x + self.drop_path(self.mlp(x))
        # print('x.shape:{}'.format(x.shape))
        # exit()

        return x


class BasicLayer(nn.Module):
    """
    Scale-Aware Semantics Extractor
    """
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None):
        super().__init__()
        self.block_num = block_num  # 默认为4

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):  # 4
            self.transformer_blocks.append(
                TFBlock(
                    embedding_dim,  # 384
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer  # nn.ReLU6
                )
            )

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class CrossModalBasicLayer(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha=0.5):
        super().__init__()
        # print('drop_path:{}'.format(drop_path))
        print('class name: {}'.format(self.__class__))

        self.block_num = block_num  # 默认为4
        self.transformer_blocks = nn.ModuleList()
        # NOTE: 第一个TF模块为CrossModalTFBlock交叉模态TF模块 用以RGB从edge中学习注意力权重
        self.transformer_blocks.append(
            CrossModalTFBlock(
                embedding_dim,  # 384
                key_dim=key_dim,  # 16
                num_heads=num_heads,  # 8
                mlp_ratio=mlp_ratio,  # 2
                attn_ratio=attn_ratio,  # 2
                drop=drop,  # 0
                # [0.025, 0.050, 0.075, 0.100]中的数值
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                act_layer=act_layer,  # nn.ReLU6
                alpha=alpha
            )
        )
        # NOTE: 承接block_num-1个普通TF模块
        for i in range(self.block_num - 1):  # 4
            self.transformer_blocks.append(
                TFBlock(
                    embedding_dim,  # 384
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[i + 1] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer  # nn.ReLU6
                )
            )

    def forward(self, rgb_feat, edge_feat):
        x = self.transformer_blocks[0](rgb_feat, edge_feat)
        # print('x shape:{} max:{} min:{}'.format(x.shape, torch.max(x), torch.min(x)))
        # exit()

        for i in range(self.block_num - 1):
            x = self.transformer_blocks[i + 1](x)

        return x


class CrossModalBasicLayerV1M1(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha=0.5):
        super().__init__()
        # print('drop_path:{}'.format(drop_path))
        print('class name: {}'.format(self.__class__))

        self.block_num = block_num  # 默认为4
        self.transformer_blocks = nn.ModuleList()
        # NOTE: 第一个TF模块为CrossModalTFBlock交叉模态TF模块 用以RGB从edge中学习注意力权重
        self.transformer_blocks.append(
            CrossModalTFBlockV1M1(
                embedding_dim,  # 384
                key_dim=key_dim,  # 16
                num_heads=num_heads,  # 8
                mlp_ratio=mlp_ratio,  # 2
                attn_ratio=attn_ratio,  # 2
                drop=drop,  # 0
                # [0.025, 0.050, 0.075, 0.100]中的数值
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                act_layer=act_layer,  # nn.ReLU6
                alpha=alpha
            )
        )
        # NOTE: 承接block_num-1个普通TF模块
        for i in range(self.block_num - 1):  # 4
            self.transformer_blocks.append(
                TFBlock(
                    embedding_dim,  # 384
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[i + 1] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer  # nn.ReLU6
                )
            )

    def forward(self, rgb_feat, edge_feat):
        x = self.transformer_blocks[0](rgb_feat, edge_feat)
        # print('x shape:{} max:{} min:{}'.format(x.shape, torch.max(x), torch.min(x)))
        # exit()

        for i in range(self.block_num - 1):
            x = self.transformer_blocks[i + 1](x)

        return x


class CrossModalBasicLayerV2(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha=0.5):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        # print('drop_path:{}'.format(drop_path))

        self.block_num = block_num  # 默认为4
        self.transformer_blocks = nn.ModuleList()
        # NOTE: 第一个TF模块为CrossModalTFBlock交叉模态TF模块 用以RGB从edge中学习注意力权重
        self.transformer_blocks.append(
            CrossModalTFBlockV2(
                embedding_dim,  # 384
                key_dim=key_dim,  # 16
                num_heads=num_heads,  # 8
                mlp_ratio=mlp_ratio,  # 2
                attn_ratio=attn_ratio,  # 2
                drop=drop,  # 0
                # [0.025, 0.050, 0.075, 0.100]中的数值
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                act_layer=act_layer,  # nn.ReLU6
                alpha=alpha
            )
        )
        # NOTE: 承接block_num-1个普通TF模块
        for i in range(self.block_num - 1):  # 4
            self.transformer_blocks.append(
                TFBlock(
                    embedding_dim,  # 384
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[i + 1] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer  # nn.ReLU6
                )
            )

    def forward(self, rgb_feat, edge_feat):
        x = self.transformer_blocks[0](rgb_feat, edge_feat)
        # print('x shape:{} max:{} min:{}'.format(x.shape, torch.max(x), torch.min(x)))
        # exit()

        for i in range(self.block_num - 1):
            x = self.transformer_blocks[i + 1](x)

        return x


class CrossModalBasicLayerV3(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, embedding_dim_list, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha=0.5):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        # print('drop_path:{}'.format(drop_path))

        self.block_num = block_num  # 默认为3
        self.transformer_blocks = nn.ModuleList()

        # NOTE: n个CrossModalTFBlock交叉模态TF模块
        for i in range(self.block_num):  # 3
            self.transformer_blocks.append(
                CrossModalTFBlockV3(
                    embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer,  # nn.ReLU6
                    alpha=alpha,
                    fm_res=32 // (2 ** i)
                )
            )

    def forward(self, rgb_feats, edge_feats):
        outputs = []
        for i in range(self.block_num):
            outputs.append(self.transformer_blocks[i](rgb_feats[i], edge_feats[i]))
            # x = self.transformer_blocks[i](rgb_feats[i], edge_feats[i])
            # outputs.append(x)
            # print('x shape:{} max:{} min:{}'.format(x.shape, torch.max(x), torch.min(x)))
            # exit()

        return outputs


# NOTE: 原版
class CrossModalBasicLayerV3M1(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, block_per_block_nums, embedding_dim_list, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha=0.5):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        print('drop_path:{}'.format(drop_path))

        self.block_num = block_num  # 默认为3
        self.transformer_blocks = nn.ModuleList()

        self.basic_tf_block_nums = block_per_block_nums - 1  # 每个block开头都是一个交叉模态TF块，后面需要承接n-1个普通TF模块

        # NOTE: n个CrossModalTFBlock交叉模态TF模块
        for i in range(self.block_num):  # 3
            self.transformer_blocks.append(
                CrossModalTFBlockV3M1(
                    embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer,  # nn.ReLU6
                    alpha=alpha,
                    fm_res=32 // (2 ** i)
                )
            )
            for j in range(self.basic_tf_block_nums):
                self.transformer_blocks.append(
                    TFBlock(
                        embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                        key_dim=key_dim,  # 16
                        num_heads=num_heads,  # 8
                        mlp_ratio=mlp_ratio,  # 2
                        attn_ratio=attn_ratio,  # 2
                        drop=drop,  # 0
                        # [0.025, 0.050, 0.075, 0.100]中的数值
                        drop_path=drop_path[j + 1] if isinstance(drop_path, list) else drop_path,
                        norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                        act_layer=act_layer  # nn.ReLU6
                    )
                )

        print('len(self.transformer_blocks):{}'.format(len(self.transformer_blocks)))

    def forward(self, rgb_feats, edge_feats):
        outputs = []
        for i in range(self.block_num):
            B, C, H, W = get_shape(rgb_feats[i])
            _, c, h, w = get_shape(edge_feats[i])
            if C != c or H != h or W != w:
                print('RGB embedding should have the same shape with edge embedding.')
                exit()

            shortcut_rgb_feat = rgb_feats[i]
            shortcut_edge_feat = edge_feats[i]
            # 交叉模态TF块 将两种特征进行融合
            x = self.transformer_blocks[i * 4](rgb_feats[i], edge_feats[i])  # 0  4  8

            for j in range(self.basic_tf_block_nums):
                x = self.transformer_blocks[(i * 4) + j + 1](x)  # 1,2,3  5,6,7  9,10,11

            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            # print('x.shape:{}'.format(x.shape))
            x = x + shortcut_rgb_feat + shortcut_edge_feat
            # print('x.shape:{}'.format(x.shape))
            outputs.append(x)

        return outputs


# NOTE: 可视化注意力图
# class CrossModalBasicLayerV3M1(nn.Module):
#     """
#     Cross-Modal Semantics Extractor
#     """
#     def __init__(self, block_num, block_per_block_nums, embedding_dim_list, key_dim, num_heads,
#                  mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
#                  norm_cfg=dict(type='BN2d', requires_grad=True),
#                  act_layer=None, alpha=0.5):
#         super().__init__()
#         print('class name: {}'.format(self.__class__))
#         print('drop_path:{}'.format(drop_path))
#
#         self.block_num = block_num  # 默认为3
#         self.transformer_blocks = nn.ModuleList()
#
#         self.basic_tf_block_nums = block_per_block_nums - 1  # 每个block开头都是一个交叉模态TF块，后面需要承接n-1个普通TF模块
#
#         # NOTE: n个CrossModalTFBlock交叉模态TF模块
#         for i in range(self.block_num):  # 3
#             self.transformer_blocks.append(
#                 CrossModalTFBlockV3M1(
#                     embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
#                     key_dim=key_dim,  # 16
#                     num_heads=num_heads,  # 8
#                     mlp_ratio=mlp_ratio,  # 2
#                     attn_ratio=attn_ratio,  # 2
#                     drop=drop,  # 0
#                     # [0.025, 0.050, 0.075, 0.100]中的数值
#                     drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
#                     norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
#                     act_layer=act_layer,  # nn.ReLU6
#                     alpha=alpha,
#                     fm_res=32 // (2 ** i)
#                 )
#             )
#             for j in range(self.basic_tf_block_nums):
#                 self.transformer_blocks.append(
#                     TFBlock(
#                         embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
#                         key_dim=key_dim,  # 16
#                         num_heads=num_heads,  # 8
#                         mlp_ratio=mlp_ratio,  # 2
#                         attn_ratio=attn_ratio,  # 2
#                         drop=drop,  # 0
#                         # [0.025, 0.050, 0.075, 0.100]中的数值
#                         drop_path=drop_path[j + 1] if isinstance(drop_path, list) else drop_path,
#                         norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
#                         act_layer=act_layer  # nn.ReLU6
#                     )
#                 )
#
#         print('len(self.transformer_blocks):{}'.format(len(self.transformer_blocks)))
#
#     def forward(self, rgb_feats, edge_feats):
#         outputs = []
#         attn_score_list = []
#         x_list = []
#         for i in range(self.block_num):
#             B, C, H, W = get_shape(rgb_feats[i])
#             _, c, h, w = get_shape(edge_feats[i])
#             if C != c or H != h or W != w:
#                 print('RGB embedding should have the same shape with edge embedding.')
#                 exit()
#
#             shortcut_rgb_feat = rgb_feats[i]
#             shortcut_edge_feat = edge_feats[i]
#             # 交叉模态TF块 将两种特征进行融合
#             x = self.transformer_blocks[i * 4](rgb_feats[i], edge_feats[i])  # 0  4  8
#
#             for j in range(self.basic_tf_block_nums):
#                 x, attn_score = self.transformer_blocks[(i * 4) + j + 1](x)  # 1,2,3  5,6,7  9,10,11
#                 attn_score_list.append(attn_score)
#                 x_list.append(x)
#
#             x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
#             # print('x.shape:{}'.format(x.shape))
#             x = x + shortcut_rgb_feat + shortcut_edge_feat
#             # print('x.shape:{}'.format(x.shape))
#             outputs.append(x)
#
#         return outputs, attn_score_list, x_list


class CrossModalBasicLayerV3M2(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, embedding_dim_list, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha=0.5):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        print('drop_path: {}'.format(drop_path))

        self.block_num = block_num  # 默认为3
        self.transformer_blocks = nn.ModuleList()

        # def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
        #              drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True), alpha=0.5,
        #              fm_res=32, sn=2, l_channels=128, g_channesl=160)

        # NOTE: n个CrossModalTFBlock交叉模态TF模块
        for i in range(self.block_num):  # 3
            self.transformer_blocks.append(
                CrossModalTFBlockV3M2(
                    embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer,  # nn.ReLU6
                    alpha=alpha,
                    fm_res=32 // (2 ** i),
                    sn=i + 1,
                    global_channels=embedding_dim_list[-1]
                )
            )

    def forward(self, rgb_feats, edge_feats):
        # for i in range(len(rgb_feats)):
        #     print('rgb_feats[{}].shape:{}'.format(i, rgb_feats[i].shape))
        # exit()
        outputs = []
        for i in range(self.block_num):
            outputs.append(self.transformer_blocks[i](rgb_feats[i], edge_feats[i], rgb_g_feat=rgb_feats[0]))
            # print('{}===================='.format(i))
            # x = self.transformer_blocks[i](rgb_feats[i], edge_feats[i])
            # outputs.append(x)
            # print('x shape:{} max:{} min:{}'.format(x.shape, torch.max(x), torch.min(x)))
            # exit()

        return outputs


class CrossModalBasicLayerV3M3(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, block_per_block_nums, embedding_dim_list, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha=0.5):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        print('drop_path:{}'.format(drop_path))

        self.block_num = block_num  # 默认为3
        self.transformer_blocks = nn.ModuleList()
        self.block_per_block_nums = block_per_block_nums
        self.basic_tf_block_nums = block_per_block_nums - 1  # 每个block开头都是一个交叉模态TF块，后面需要承接n-1个普通TF模块

        # NOTE: n个CrossModalTFBlock交叉模态TF模块
        for i in range(self.block_num):  # 3
            self.transformer_blocks.append(
                CrossModalTFBlockV3M1(
                    embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer,  # nn.ReLU6
                    alpha=alpha,
                    fm_res=32 // (2 ** i)
                )
            )
            for j in range(self.basic_tf_block_nums):
                self.transformer_blocks.append(
                    TFBlock(
                        embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                        key_dim=key_dim,  # 16
                        num_heads=num_heads,  # 8
                        mlp_ratio=mlp_ratio,  # 2
                        attn_ratio=attn_ratio,  # 2
                        drop=drop,  # 0
                        # [0.025, 0.050, 0.075, 0.100]中的数值
                        drop_path=drop_path[j + 1] if isinstance(drop_path, list) else drop_path,
                        norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                        act_layer=act_layer  # nn.ReLU6
                    )
                )

        print('len(self.transformer_blocks):{}'.format(len(self.transformer_blocks)))

    def forward(self, rgb_feats, edge_feats):
        outputs = []
        for i in range(self.block_num):
            B, C, H, W = get_shape(rgb_feats[i])
            _, c, h, w = get_shape(edge_feats[i])
            if C != c or H != h or W != w:
                print('RGB embedding should have the same shape with edge embedding.')
                exit()

            shortcut_rgb_feat = rgb_feats[i]
            shortcut_edge_feat = edge_feats[i]
            # 交叉模态TF块 将两种特征进行融合
            x = self.transformer_blocks[i * self.block_per_block_nums](rgb_feats[i], edge_feats[i])  # 0  4  8

            for j in range(self.basic_tf_block_nums):
                x = self.transformer_blocks[(i * self.block_per_block_nums) + j + 1](x)  # 1,2,3  5,6,7  9,10,11

            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x + shortcut_rgb_feat + shortcut_edge_feat
            outputs.append(x)

        return outputs


class CrossModalBasicLayerV3M4(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, block_per_block_nums, embedding_dim_list, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha='learnable'):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        print('drop_path:{}'.format(drop_path))

        self.block_num = block_num  # 默认为3
        self.transformer_blocks = nn.ModuleList()

        self.basic_tf_block_nums = block_per_block_nums - 1  # 每个block开头都是一个交叉模态TF块，后面需要承接n-1个普通TF模块

        # NOTE: 将可学习的比例系数初始化为了0.5
        # self.alpha1 = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        # self.alpha_list = []
        # self.alpha_list.append(self.alpha1)
        # self.alpha_list.append(self.alpha1)
        # self.alpha_list.append(self.alpha1)
        self.alpha_list = []
        if alpha == 'learnable':
            for i in range(self.block_num):
                self.alpha_list.append(torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True))
        elif type(alpha) == list:
            for i in range(self.block_num):
                self.alpha_list.append(alpha[i])

        for i in range(len(self.alpha_list)):
            print('self.alpha_list[{}]:{}'.format(i, self.alpha_list[i]))

        # NOTE: n个CrossModalTFBlock交叉模态TF模块
        for i in range(self.block_num):  # 3
            self.transformer_blocks.append(
                CrossModalTFBlockV3M1(
                    embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer,  # nn.ReLU6
                    alpha=self.alpha_list[i],
                    fm_res=32 // (2 ** i)
                )
            )
            for j in range(self.basic_tf_block_nums):
                self.transformer_blocks.append(
                    TFBlock(
                        embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                        key_dim=key_dim,  # 16
                        num_heads=num_heads,  # 8
                        mlp_ratio=mlp_ratio,  # 2
                        attn_ratio=attn_ratio,  # 2
                        drop=drop,  # 0
                        # [0.025, 0.050, 0.075, 0.100]中的数值
                        drop_path=drop_path[j + 1] if isinstance(drop_path, list) else drop_path,
                        norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                        act_layer=act_layer  # nn.ReLU6
                    )
                )

        print('len(self.transformer_blocks):{}'.format(len(self.transformer_blocks)))

    def forward(self, rgb_feats, edge_feats):
        outputs = []
        for i in range(self.block_num):
            B, C, H, W = get_shape(rgb_feats[i])
            _, c, h, w = get_shape(edge_feats[i])
            if C != c or H != h or W != w:
                print('RGB embedding should have the same shape with edge embedding.')
                exit()

            # print('block_num:{}'.format(i))

            shortcut_rgb_feat = rgb_feats[i]
            shortcut_edge_feat = edge_feats[i]
            # 交叉模态TF块 将两种特征进行融合
            x = self.transformer_blocks[i * 4](rgb_feats[i], edge_feats[i])  # 0  4  8

            for j in range(self.basic_tf_block_nums):
                x = self.transformer_blocks[(i * 4) + j + 1](x)  # 1,2,3  5,6,7  9,10,11

            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            # print('x.shape:{}'.format(x.shape))
            x = x + shortcut_rgb_feat + shortcut_edge_feat
            # print('x.shape:{}'.format(x.shape))
            outputs.append(x)

        return outputs


class CrossModalBasicLayerV3M5(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, block_per_block_nums, embedding_dim_list, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha=0.5):
        super().__init__()
        print('class name: {}'.format(self.__class__))
        print('drop_path:{}'.format(drop_path))

        self.block_num = block_num  # 默认为3
        self.transformer_blocks = nn.ModuleList()
        self.block_per_block_nums = block_per_block_nums
        relay_tf_block_nums = block_per_block_nums - 1  # 每个block开头都是一个交叉模态TF块，后面需要承接n-1个普通TF模块

        # NOTE: n个CrossModalTFBlock交叉模态TF模块
        for i in range(self.block_num):  # 3
            self.transformer_blocks.append(
                CrossModalTFBlockV3M1(
                    embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer,  # nn.ReLU6
                    alpha=alpha,
                    fm_res=32 // (2 ** i)
                )
            )
            for j in range(relay_tf_block_nums):
                self.transformer_blocks.append(
                    RelayCrossModalTFBlockV1(
                        embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                        key_dim=key_dim,  # 16
                        num_heads=num_heads,  # 8
                        mlp_ratio=mlp_ratio,  # 2
                        attn_ratio=attn_ratio,  # 2
                        drop=drop,  # 0
                        # [0.025, 0.050, 0.075, 0.100]中的数值
                        drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                        norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                        act_layer=act_layer,  # nn.ReLU6
                        alpha=alpha,
                        fm_res=32 // (2 ** i)
                    )
                )

        print('len(self.transformer_blocks):{}'.format(len(self.transformer_blocks)))

    def forward(self, rgb_feats, edge_feats):
        outputs = []
        for i in range(self.block_num):
            B, C, H, W = get_shape(rgb_feats[i])
            _, c, h, w = get_shape(edge_feats[i])
            if C != c or H != h or W != w:
                print('RGB embedding should have the same shape with edge embedding.')
                exit()

            shortcut_rgb_feat = rgb_feats[i]
            shortcut_edge_feat = edge_feats[i]
            # 交叉模态TF块 将两种特征进行融合
            for j in range(self.block_per_block_nums):  # 每个block中含有4个交叉TF模块
                if j == 0:
                    # print('serial number:{}'.format(i * 4))
                    x = self.transformer_blocks[i * 4](rgb_feats[i], edge_feats[i])  # 0  4  8
                else:
                    # print('serial number:{}'.format(i * 4 + j))
                    x = self.transformer_blocks[i * 4 + j](x, edge_feats[i])  # 0  4  8

            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            # print('x.shape:{}'.format(x.shape))
            x = x + shortcut_rgb_feat + shortcut_edge_feat
            # print('x.shape:{}'.format(x.shape))
            outputs.append(x)

        return outputs


class CrossModalBasicLayerV3M6(nn.Module):
    """
    Cross-Modal Semantics Extractor
    """
    def __init__(self, block_num, block_per_block_nums, embedding_dim_list, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None, alpha='learnable'):
        super().__init__()
        self.block_num = block_num  # 默认为3
        self.transformer_blocks = nn.ModuleList()
        self.block_per_block_nums = block_per_block_nums
        relay_tf_block_nums = block_per_block_nums - 1  # 每个block开头都是一个交叉模态TF块，后面需要承接n-1个普通TF模块

        # 初始化为0.5
        self.alpha_list = []
        for i in range(block_num * block_per_block_nums):  # 3 * 4 = 12个子模块都需要可学习的参数
            self.alpha_list.append(torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True))

        # NOTE: n个CrossModalTFBlock交叉模态TF模块
        for i in range(self.block_num):  # 3
            print('sn:{}'.format(i * 4))
            self.transformer_blocks.append(
                CrossModalTFBlockV3M1(
                    embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                    key_dim=key_dim,  # 16
                    num_heads=num_heads,  # 8
                    mlp_ratio=mlp_ratio,  # 2
                    attn_ratio=attn_ratio,  # 2
                    drop=drop,  # 0
                    # [0.025, 0.050, 0.075, 0.100]中的数值
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                    act_layer=act_layer,  # nn.ReLU6
                    alpha=self.alpha_list[i * 4],
                    fm_res=32 // (2 ** i)
                )
            )
            for j in range(relay_tf_block_nums):
                print('sn:{}'.format(i * 4 + j + 1))
                self.transformer_blocks.append(
                    RelayCrossModalTFBlockV1(
                        embedding_dim_list[-(1 + i)],  # [64, 128, 160] 倒序取出列表中的数值
                        key_dim=key_dim,  # 16
                        num_heads=num_heads,  # 8
                        mlp_ratio=mlp_ratio,  # 2
                        attn_ratio=attn_ratio,  # 2
                        drop=drop,  # 0
                        # [0.025, 0.050, 0.075, 0.100]中的数值
                        drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                        norm_cfg=norm_cfg,  # dict(type='BN2d', requires_grad=True)
                        act_layer=act_layer,  # nn.ReLU6
                        alpha=self.alpha_list[i * 4 + j + 1],
                        fm_res=32 // (2 ** i)
                    )
                )

        print('len(self.transformer_blocks):{}'.format(len(self.transformer_blocks)))

    def forward(self, rgb_feats, edge_feats):
        outputs = []
        for i in range(self.block_num):
            B, C, H, W = get_shape(rgb_feats[i])
            _, c, h, w = get_shape(edge_feats[i])
            if C != c or H != h or W != w:
                print('RGB embedding should have the same shape with edge embedding.')
                exit()

            shortcut_rgb_feat = rgb_feats[i]
            shortcut_edge_feat = edge_feats[i]
            # 交叉模态TF块 将两种特征进行融合
            for j in range(self.block_per_block_nums):  # 每个block中含有4个交叉TF模块
                if j == 0:
                    # print('serial number:{}'.format(i * 4))
                    x = self.transformer_blocks[i * 4](rgb_feats[i], edge_feats[i])  # 0  4  8
                else:
                    # print('serial number:{}'.format(i * 4 + j))
                    x = self.transformer_blocks[i * 4 + j](x, edge_feats[i])  # 0  4  8

            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x + shortcut_rgb_feat + shortcut_edge_feat
            outputs.append(x)

        return outputs


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SemanticInjectionModule(nn.Module):
    """
    NOTE: 原始的类名字为--InjectionMultiSum
    Semantic Injection Module
    """
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN2d', requires_grad=True),
            activations=None,
    ) -> None:
        super(SemanticInjectionModule, self).__init__()
        self.norm_cfg = norm_cfg

        # 默认 inp = oup 不改变通道数
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        # x_l: [b, 64, h/8, w/8]     x_g: [b, 64, h/64, w/64]
        # x_l: [b, 128, h/16, w/16]  x_g: [b, 128, h/64, w/64]
        # x_l: [b, 160, h/32, w/32]  x_g: [b, 160, h/64, w/64]
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)  # [b, 128, h/x, w/x]

        global_act = self.global_act(x_g)  # [b, 128, h/64, w/64]
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)  # [b, 128, h/x, w/x]

        global_feat = self.global_embedding(x_g)  # [b, 128, h/64, w/64]
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)  # [b, 128, h/x, w/x]

        out = local_feat * sig_act + global_feat  # [b, 128, h/x, w/x]
        return out


# class PyramidPoolAgg(nn.Module):
#     """
#     NOTE: LIAM Modified
#     """
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, inputs):
#         B, C, H, W = get_shape(inputs[-1])  # encoder最后一层的输出形状[b, c, h/8, w/8]
#         H = H // 4  # 进一步下采样h/32
#         W = W // 4  # 进一步下采样w/32
#         # 输出的形状仍然是[b, c, h/64, w/64] encoder四层的输出在通道维度上cat 分辨率都是原始的1/64
#         return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])  # encoder最后一层的输出形状[b, c, h/32, w/32]
        H = (H - 1) // self.stride + 1  # 进一步下采样h/64
        W = (W - 1) // self.stride + 1  # 进一步下采样w/64
        # 输出的形状仍然是[b, c, h/64, w/64] encoder四层的输出在通道维度上cat 分辨率都是原始的1/64
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class LTAM(nn.Module):
    """
    Local Token Aggregation Module
    """
    def __init__(self,
                 rgb_feats,
                 edge_feats,
                 out_feats):
        super(LTAM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(rgb_feats + edge_feats, edge_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(edge_feats),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(edge_feats, out_feats, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb_feat, edge_feat):
        rgb_feat = F.interpolate(rgb_feat, size=edge_feat.size()[2:], mode='bilinear')
        shortcut_edge = edge_feat

        fused_feat = torch.cat((rgb_feat, edge_feat), dim=1)
        fused_feat = self.conv1(fused_feat)
        fused_feat = fused_feat + shortcut_edge
        fused_feat = self.conv2(fused_feat)

        return fused_feat


class LTG(nn.Module):
    """
    Loacl Token Generator
    """
    def __init__(self):
        super(LTG, self).__init__()

        self.block1 = LTAM(128, 128, 128)
        self.block2 = LTAM(64, 64, 64)
        self.block3 = LTAM(32, 32, 32)

    def forward(self, rgb_features, edge_features):
        # print('rgb_features[-1].shape:', rgb_features[-1].shape)
        # print('rgb_features[-3].shape:', rgb_features[-3].shape)
        # print('rgb_features[-5].shape:', rgb_features[-5].shape)
        # print('edge_features[-1].shape:', edge_features[-1].shape)
        # print('edge_features[-2].shape:', edge_features[-2].shape)
        # print('edge_features[-3].shape:', edge_features[-3].shape)
        local_tokens = []
        local_tokens.append(self.block1(rgb_features[-1], edge_features[-1]))
        local_tokens.append(self.block2(rgb_features[-3], edge_features[-2]))
        local_tokens.append(self.block3(rgb_features[-5], edge_features[-3]))
        return local_tokens


class GTG(nn.Module):
    """
    Global Token Generator
    """
    def __init__(self):
        super(GTG, self).__init__()

    def forward(self, local_tokens):
        b, c, h, w = local_tokens[0].shape
        return torch.cat([nn.functional.adaptive_avg_pool2d(lt, (h, w)) for lt in local_tokens], dim=1)


class UpsamplingModuleV1(nn.Module):
    """
    Upsampling Module
    """
    def __init__(self, lr_feats, out_feats, hr_feats=None, up_scale=2):
        super(UpsamplingModuleV1, self).__init__()
        # self.h_feats = h_feats
        self.up_scale = up_scale

        self.conv = nn.Sequential(
            nn.Conv2d(lr_feats, out_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_feats),
            nn.ReLU6(inplace=True),
        )

    def forward(self, lr_token, hr_token=None):
        if hr_token is not None:
            x = lr_token + hr_token
        else:
            x = lr_token

        x = F.interpolate(x, scale_factor=self.up_scale, mode='bilinear')
        x = self.conv(x)

        return x


class RRDecoderV1(nn.Module):
    """
    Resolution Recovery Decoder
    """
    def __init__(self):
        super(RRDecoderV1, self).__init__()

        self.upsampling_1 = UpsamplingModuleV1(128, 128)
        self.upsampling_2 = UpsamplingModuleV1(128, 128, hr_feats=128)
        self.upsampling_3 = UpsamplingModuleV1(128, 1, hr_feats=128, up_scale=4)

    def forward(self, injected_tokens):
        x = self.upsampling_1(injected_tokens[0])
        x = self.upsampling_2(x, injected_tokens[1])
        x = self.upsampling_3(x, injected_tokens[2])
        return x




















