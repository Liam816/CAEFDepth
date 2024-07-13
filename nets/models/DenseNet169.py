import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)
        # print('class UpSample initializing.')

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(
            self.convA(torch.cat([up_x, concat_with], dim=1))))  # 论文中appendix部分提到：只对convB的结果进行leakyrelu，而不对convA进行


class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width=1.0):
        super(Decoder, self).__init__()

        # print('class Decoder initializing.')

        features = int(num_features * decoder_width)
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)  # [1664, 1664, h, w]

        self.up1 = UpSample(skip_input=features // 1 + 256, output_features=features // 2)  # 第一个上采样层的输入包括：上一层卷积结果+encoder倒数第二层结果(也即256个通道)
        self.up2 = UpSample(skip_input=features // 2 + 128, output_features=features // 4)
        self.up3 = UpSample(skip_input=features // 4 + 64, output_features=features // 8)
        self.up4 = UpSample(skip_input=features // 8 + 64, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        # relu0   pool0  transition1  transition2  norm5
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[12]
        x_d0 = self.conv2(F.relu(x_block4))  # x_block4: [bs, 1664, 15, 20]  x_d0: [bs, 1664, 15, 20]

        x_d1 = self.up1(x_d0, x_block3)  # input_features = 1664, 256  output_feature = 832  x_d1: [bs, 256, 30, 40]   x_block3: [bs, 256, 30, 40]
        x_d2 = self.up2(x_d1, x_block2)  # input_features = 832, 128   output_feature = 416  x_d2: [bs, 416, 60, 80]   x_block2: [bs, 128, 60, 80]
        x_d3 = self.up3(x_d2, x_block1)  # input_features = 416, 64    output_feature = 208  x_d3: [bs, 208, 120, 160]   x_block1: [bs, 64, 120, 160]
        x_d4 = self.up4(x_d3, x_block0)  # input_features = 208, 64    output_feature = 104  x_d4: [bs, 104, 240, 320]   x_block0: [bs, 64, 240, 320]
        return self.conv3(x_d4)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.original_model = models.densenet169( pretrained=False )
        self.original_model = models.densenet169(weights=None)
        # print('class Encoder initializing.')

    def forward(self, x):
        features = [x]
        # k:网络每个模块的名称 v:网络每个模块，可以单独用来做前向传播使用 k:key v:value
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))  # 把每层的特征图保存到一个list中
        # featyres = [x, conv0, norm0, relu0, pool0, denseblock1, transition1, denseblock2, transition2, denseblock3, transition3, denseblock4, norm5]
        return features


class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        # print('class PTModel initializing.')
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


