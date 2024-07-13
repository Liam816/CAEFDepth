import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleAttention(nn.Module):
    """
    Cross-Scale Attention CSA
    """
    def __init__(self, hr_feats, lr_feats, hidden_feats):
        super(CrossScaleAttention, self).__init__()
        self.hidden_feats = hidden_feats
        self.proj1 = nn.Linear(hr_feats, hidden_feats, bias=False)
        self.proj2 = nn.Linear(lr_feats, hidden_feats, bias=False)
        self.proj_final = nn.Linear(hidden_feats, hr_feats, bias=False)

    def forward(self, hr_feat, lr_feat):
        b, c1, h1, w1 = hr_feat.shape
        _, c2, h2, w2 = lr_feat.shape

        x1 = hr_feat.reshape(b, c1, h1 * w1).permute(0, 2, 1)  # [b, h1 * w1, c1]
        x1 = self.proj1(x1)  # [b, h1 * w1, c]
        # print('x1.shape:{}'.format(x1.shape))

        x2 = lr_feat.reshape(b, c2, h2 * w2).permute(0, 2, 1)  # [b, h2 * w2, c2]
        x2 = self.proj2(x2)  # [b, h2 * w2, c]
        vv = x2  # [b, h2 * w2, c]
        # print('vv.shape:{}'.format(vv.shape))
        x2 = x2.permute(0, 2, 1)  # [b, c, h2 * w2]
        # print('x2.shape:{}'.format(x2.shape))

        attn = torch.matmul(x1, x2).reshape(b, (h1 * w1) * (h2 * w2))  # 把二维特征图展开成一维向量
        attn = attn.softmax(dim=-1).reshape(b, h1 * w1, h2 * w2)
        # print('attn.shape:{}'.format(attn.shape))

        xx = torch.matmul(attn, vv)  # [b, h1 * w1, c]
        xx = xx + x1  # [b, h1 * w1, c]
        # xx = self.proj_final(xx)  # [b, h1 * w1, c1]
        # print('xx.shape:{}'.format(xx.shape))
        xx = xx.permute(0, 2, 1).reshape(b, self.hidden_feats, h1, w1)  # [b, c, h1, w1]
        # print('xx.shape:{}'.format(xx.shape))

        return xx


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
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


class MultiScaleRefinement(nn.Module):
    """
    Multi-Scale Refinement MSR
    """
    def __init__(self, dim, hidden_dim, out_dim, is_last=False):
        super(MultiScaleRefinement, self).__init__()
        self.out_dim = out_dim
        self.is_last = is_last
        # self.mlp1 = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=nn.ReLU6, drop=0.)
        # self.mlp2 = Mlp(in_features=hidden_dim, hidden_features=out_dim, act_layer=nn.ReLU6, drop=0.)
        self.mlp = Mlp(in_features=dim, hidden_features=out_dim * 2, out_features=out_dim, act_layer=nn.ReLU6, drop=0.)

    def forward(self, hr_feat, lr_feat=None):
        if lr_feat is not None:
            lr_feat = F.interpolate(lr_feat, scale_factor=2.0, mode='bilinear')
            x = hr_feat + lr_feat
            b, c, h, w = x.shape
            x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h * w, c]
            # print('x.shape:{}'.format(x.shape))
            x = self.mlp(x)
            # print('x.shape:{}'.format(x.shape))
            x = x.permute(0, 2, 1).reshape(b, self.out_dim, h, w)
        else:
            x = F.interpolate(hr_feat, scale_factor=2.0, mode='bilinear')
            b, c, h, w = x.shape
            x = x.reshape(b, c, h * w).permute(0, 2, 1)  # [b, h * w, c]
            x = self.mlp(x)
            x = x.permute(0, 2, 1).reshape(b, self.out_dim, h, w)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cas1 = CrossScaleAttention(128, 160, 64)
        self.cas2 = CrossScaleAttention(64, 64, 64)
        self.cas3 = CrossScaleAttention(32, 64, 32)

        self.msr1 = MultiScaleRefinement(64, None, out_dim=32)
        self.msr2 = MultiScaleRefinement(32, None, out_dim=16)
        self.msr3 = MultiScaleRefinement(16, None, out_dim=8)
        self.msr4 = MultiScaleRefinement(8, None, out_dim=1)

    def forward(self, features):
        x1 = self.cas1(features[1], features[0])
        x2 = self.cas2(features[2], x1)
        x3 = self.cas3(features[3], x2)
        print('x1.shape:{}'.format(x1.shape))
        print('x2.shape:{}'.format(x2.shape))
        y1 = self.msr1(x2, x1)
        y2 = self.msr2(x3, y1)
        y3 = self.msr3(y2)
        y4 = self.msr4(y3)
        print('y4.shape:{}'.format(y4.shape))

        return y4


if __name__ == '__main__':

    x1 = torch.randn(size=(8, 160, 15, 20))
    x2 = torch.randn(size=(8, 128, 30, 40))
    x3 = torch.randn(size=(8, 64, 60, 80))
    x4 = torch.randn(size=(8, 32, 120, 160))

    features = [x1, x2, x3, x4]

    model = Net()
    res = model.forward(features)
    exit()

    model = CrossScaleAttention(64, 128, 32)
    res = model.forward(x1, x2)


