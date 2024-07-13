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
class TFDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(TFDropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TFPatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=(480, 640), patch_size=16, in_c=3, norm_layer=None):
        super().__init__()
        embed_dim = patch_size * patch_size * in_c  # 默认为16*16*3=768
        # img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # (480/16, 640/16)=(30, 40)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  # 默认为None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # proj: [B, C, H, W] -> [B, embed_dim, grid_size[0], grid_size[1]]
        # flatten: [B, embed_dim, grid_size[0], grid_size[1]] -> [B, embed_dim, grid_size[0]*grid_size[1]]
        #                                                      = [B, embed_dim, num_patches]
        # transpose: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# NOTE: PPT P42 所使用的unfold操作
class PatchEmbedUnfold(nn.Module):
    def __init__(self, feature_map_size=(30, 40), patch_size=1, in_c=32):
        super(PatchEmbedUnfold, self).__init__()
        self.fms = feature_map_size
        self.ps = patch_size
        self.in_c = in_c
        # print('self.fms:{} self.ps:{}'.format(self.fms, self.ps))

    def forward(self, patch_embedding):
        B, patch_nums, embedding_dims = patch_embedding.size()
        c, h, w = embedding_dims // (self.ps ** 2), self.fms[0] // self.ps, self.fms[1] // self.ps
        # print('c:{} h:{} w:{}'.format(c, h, w))

        # embedding_dims = ps * ps * channels
        # patch_nums = (fms_h/ps) * (fms_w/ps)
        x = patch_embedding.permute(0, 2, 1)  # [b, embedding_dims, patch_nums]
        x = x.contiguous().view(B, self.ps ** 2, c, h, w)
        # print('x.shape:{}'.format(x.shape))
        x = torch.mean(x, dim=1)
        # print('x.shape:{}'.format(x.shape))
        return x


# NOTE: 展开成压缩成embedding的特征图的尺寸
# class PatchEmbedUnfold(nn.Module):
#     def __init__(self, feature_map_size=(30, 40), patch_size=1, in_c=32):
#         super(PatchEmbedUnfold, self).__init__()
#         self.fms = feature_map_size
#         self.ps = patch_size
#         self.in_c = in_c
#         # print('self.fms:{} self.ps:{}'.format(self.fms, self.ps))
#
#     def forward(self, patch_embedding):
#         B, patch_nums, embedding_dims = patch_embedding.size()
#         C = embedding_dims // (self.ps ** 2)
#         # print('c:{} h:{} w:{}'.format(c, h, w))
#
#         # embedding_dims = ps * ps * channels
#         # patch_nums = (fms_h/ps) * (fms_w/ps)
#         # [b, embedding_dims, patch_nums]
#         # [b, ps * ps * channels, (fms_h/ps) * (fms_w/ps)]
#         x = patch_embedding.permute(0, 2, 1)
#         # x = x.reshape(B, C, self.ps * self.ps, patch_nums)
#         x = x.contiguous().view(B, C, self.fms[0], self.fms[1])
#         # print('x.shape:{}'.format(x.shape))
#         return x


class Attention(nn.Module):
    """
    NOTE: from bilibili code
    """
    def __init__(self,
                 dim,   # 输入patch_embedding也即token的dim embed_dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 visualization=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 将输入的patch_embedding映射成qkv三种值 博客中的a->q, k, v
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.vis = visualization

    def forward(self, x):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x.shape  # b, 300, 10

        # qkv(): -> [B, num_patches, 3 * total_embed_dim]
        # reshape: -> [B, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, B, num_heads, num_patches, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, num_heads, num_patches, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch script happy (cannot use tensor as tuple)

        # transpose: -> [B, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [B, num_heads, num_patches, num_patches]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MyAttention(nn.Module):
    """
    NOTE: LIAM 修改的
    """
    def __init__(self,
                 dim,   # 输入patch_embedding也即token的dim embed_dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 visualization=False):
        super(MyAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 将输入的patch_embedding映射成qkv三种值 博客中的a->q, k, v
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.vis = visualization

    def forward(self, x):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x.shape  # b, 300, 10

        # qkv(): -> [B, num_patches, 3 * total_embed_dim]
        # reshape: -> [B, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, B, num_heads, num_patches, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, num_heads, num_patches, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch script happy (cannot use tensor as tuple)

        # transpose: -> [B, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [B, num_heads, num_patches, num_patches]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # NOTE: 注意在softmax之前还是之后给出这个相似性分数，因为后续需要将Backbone和EdgeNet输出的qk相加，
        #  softmax会将qk分数直接映射到[0, 1]之间
        similarity_score = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.vis is True:
            return x, attn
        else:
            return x


class SimilarityQK(nn.Module):
    def __init__(self,
                 dim,   # 输入patch_embedding也即token的dim embed_dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.):
        super(SimilarityQK, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 将输入的patch_embedding映射成qkv三种值 博客中的a->q, k, v
        self.attn_drop = nn.Dropout(attn_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x.shape  # b, 300, 10

        # qk(): -> [B, num_patches, 2 * total_embed_dim]
        # reshape: -> [B, num_patches, 2, num_heads, embed_dim_per_head]
        # permute: -> [2, B, num_heads, num_patches, embed_dim_per_head]
        qkv = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, num_heads, num_patches, embed_dim_per_head]
        q, k = qkv[0], qkv[1]

        # transpose: -> [B, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [B, num_heads, num_patches, num_patches]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # NOTE: 注意在softmax之前还是之后给出这个相似性分数，因为后续需要将Backbone和EdgeNet输出的qk相加，
        # NOTE: softmax会将qk分数直接映射到[0, 1]之间
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        return attn


class GetV(nn.Module):
    def __init__(self,
                 dim,   # 输入patch_embedding也即token的dim embed_dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None):
        super(GetV, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.v = nn.Linear(dim, dim * 1, bias=qkv_bias)  # 将输入的patch_embedding映射成qkv三种值 博客中的a->q, k, v

    def forward(self, x):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x.shape  # b, 300, 10

        # v(): -> [B, num_patches, 1 * total_embed_dim]
        # reshape: -> [B, num_patches, 1, num_heads, embed_dim_per_head]
        # permute: -> [1, B, num_heads, num_patches, embed_dim_per_head]
        v = self.v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, num_heads, num_patches, embed_dim_per_head]
        v = v[0]

        return v


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TFBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(TFBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = TFDropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # x = self.norm1(x)  # [8, 197, 768]
        # # print('x after norm1:', x.shape)
        # x = self.attn(x)  # [8, 197, 768]
        # # print('x after attn:', x.shape)
        #
        # x = x + self.drop_path(x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x






