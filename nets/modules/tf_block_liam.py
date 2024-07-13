import torch
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
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


# NOTE: 原来也叫 DropPath
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, in_c=3):
        super().__init__()

        self.proj = nn.Linear(in_c, in_c, bias=False)

    def forward(self, x):
        # flatten: [B, C, H, W] -> [B, C, H * W]
        # permute: [B, C, H * W] -> [B, H * W, C]
        # x = x.flatten(2).permute(1, 2)
        x = x.flatten(2).transpose(1, 2)
        # print('x.shape:{}'.format(x.shape))
        x = self.proj(x)  # [B, H * W, C]
        # print('x.shape:{}'.format(x.shape))
        return x


class LightWeightMHSA(nn.Module):
    """
    NOTE: 权重矩阵不再是原来的方阵
    """
    def __init__(self,
                 input_dim,   # 输入patch_embedding也即token的dim embed_dim
                 num_heads=2,
                 qkv_dim_per_head=16,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 visualization=False):
        super(LightWeightMHSA, self).__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or qkv_dim_per_head ** -0.5
        self.qkv_dim_per_head = qkv_dim_per_head
        hidden_dim = num_heads * qkv_dim_per_head
        # 将输入的patch_embedding映射成qkv三种值 博客中的a->q, k, v
        self.qkv = nn.Linear(input_dim, 3 * hidden_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(hidden_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.vis = visualization

    def forward(self, x):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x.shape  # b, 300, 10

        # qkv(): -> [B, num_patches, hidden_dim]
        # reshape: -> [B, num_patches, 3, num_heads, qkv_dim_per_head]
        # permute: -> [3, B, num_heads, num_patches, qkv_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.qkv_dim_per_head).permute(2, 0, 3, 1, 4)
        # [B, num_heads, num_patches, qkv_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch script happy (cannot use tensor as tuple)

        # transpose: -> [B, num_heads, qkv_dim_per_head, num_patches]
        # @: multiply -> [B, num_heads, num_patches, num_patches]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches, qkv_dim_per_head]
        # transpose: -> [batch_size, num_patches, num_heads, qkv_dim_per_head]
        # reshape: -> [batch_size, num_patches, hidden_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # proj: -> [batch_size, num_patches, input_dim]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttnBlock(nn.Module):
    def __init__(self,
                 input_dim=32,  # 输入patch_embedding也即token的dim embed_dim
                 num_heads=2,
                 qkv_dim_per_head=16,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 drop_path_ratio=0.,  # 根据AttnBlock的数量逐层增加
                 visualization=False):
        super(AttnBlock, self).__init__()

        # self.h, self.w = feature_map_shape[0], feature_map_shape[1]
        self.to_embed = PatchEmbed(input_dim)
        self.attn = LightWeightMHSA(input_dim,
                                    num_heads,
                                    qkv_dim_per_head,
                                    qkv_bias,
                                    qk_scale,
                                    attn_drop_ratio,
                                    proj_drop_ratio,
                                    visualization)
        self.drop_path = DropPath(drop_path_ratio)

    def forward(self, x):
        b, c, h, w = x.shape
        embedding = self.to_embed(x)
        shortcut = embedding
        embedding = self.attn(embedding)  # [b, patch_nums, input_dims]
        embedding = shortcut + self.drop_path(embedding)

        assert embedding.shape[1] == h * w, 'The number of patches should be equal to h * w!'
        assert embedding.shape[2] == c, 'The dims of embedding should be equal to c'

        # embedding -> feature map
        y = embedding.transpose(1, 2).reshape(b, c, h, w)

        return y











