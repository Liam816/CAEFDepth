import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()  # 执行父类的__init__方法
        # Nh为高度方向尺寸，Nw为宽度方向尺寸，s为stride
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = [(Nh-1)/s]+1 * [(Nw-1)/s]+1
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = [(Nh-1)]+1 * [(Nw-1)]+1
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = [(Nh-1)]+1 * [(Nw-1)]+1
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        # print('step1:{}'.format(Y.shape))
        Y = self.bn2(self.conv2(Y))
        # print('step2:{}'.format(Y.shape))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


# Resnet18
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blks = []
        self.b1 = nn.Sequential(
            # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = [(Nh-1)/2]+1 * [(Nw-1)/2]+1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = [(Nh-1)/2]+1 * [(Nw-1)/2]+1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.blks.append(self.b1)

        # 一个残差网络块包含任意个残差块
        def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
            blk = []
            for i in range(num_residuals):
                if i==0 and not first_block:
                    # 每一个残差块将高宽减半，输出通道数翻倍
                    blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
                else:
                    # 第一个残差网络块的每个残差块输入和输出通道数一致，且高宽不变
                    blk.append(Residual(num_channels, num_channels))
            return blk

        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.blks.append(self.b2)
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.blks.append(self.b3)
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.blks.append(self.b4)
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.blks.append(self.b5)

        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )
        self.blks.append(self.b6)

        # self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, self.b6)
        # self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5,
        #                          nn.AdaptiveAvgPool2d((1, 1)),
        #                          nn.Flatten(),
        #                          nn.Linear(512, 10)
        #                          )

    def forward(self, X):
        features = [X]
        for i in range(len(self.blks)):
            features.append(self.blks[i](features[-1]))

        # Y = self.b1(X)
        # print('step1:{}'.format(Y.shape))
        # Y = self.b2(Y)
        # print('step2:{}'.format(Y.shape))
        # Y = self.b3(Y)
        # print('step3:{}'.format(Y.shape))
        # Y = self.b4(Y)
        # print('step4:{}'.format(Y.shape))
        # Y = self.b5(Y)
        # print('step5:{}'.format(Y.shape))
        # Y = self.b6(Y)
        # print('step6:{}'.format(Y.shape))
        #
        # # Y = self.net(X)
        # return Y

        return features


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(self).__init__()
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        # self.leakyreluA = nn.LeakyReLU(0.2)
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)
        # print('class UpSample initializing.')

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(
            self.convA(torch.cat([up_x, concat_with], dim=1))))  # 论文中appendix部分提到：只对convB的结果进行leakyrelu，而不对convA进行


class Decoder(nn.Module):
    def __init__(self, num_features=512, decoder_width=1.0):
        super(self).__init__()

        features = int(num_features * decoder_width)  # 512
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)  # [bs, 512, h, w]
        # 把UpSample封装成nn.Module!!!!!!!!!!!!!!!!!!!!!
        self.up1 = UpSample(skip_input=features // 1 + 256, output_features=features // 2)  # 联结b4 output_channels: 256
        self.up2 = UpSample(skip_input=features // 2 + 128, output_features=features // 4)  # 联结b3 output_channels: 128
        self.up3 = UpSample(skip_input=features // 4 + 64, output_features=features // 8)   # 联结b2 output_channels: 64
        self.up4 = UpSample(skip_input=features // 8 + 64, output_features=features // 16)  # 联结b1 output_channels: 32
        # 输出形状为[(Nh-k+2p)/s]+1 * [(Nw-k+2p)/s]+1 = Nh * Nw
        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)  # output_channels: 1

    def forward(self, features):
        # b1      b2        b3        b4        b5
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[1], features[2], features[3], features[4], features[5]
        x_d0 = self.conv2(F.relu(x_block4))  # input: [bs, 512, 15, 20], output: [bs, 512, 15, 20]
        x_d1 = self.up1(x_d0, x_block3)  # input: [bs, 512+256, 30, 40], output: [bs, 256, 30, 40]
        x_d2 = self.up2(x_d1, x_block2)  # input: [bs, 256+128, 60, 80], output: [bs, 128, 60, 80]
        x_d3 = self.up3(x_d2, x_block1)  # input: [bs, 128+64, 120, 160], output: [bs, 64, 120, 160]
        x_d4 = self.up4(x_d3, x_block0)  # input: [bs, 64+64, 120, 160], output: [bs, 32, 120, 160]
        return self.conv3(x_d4)  # output: [bs, 1, 120, 160]


class Resnet18Model(nn.Module):
    def __init__(self):
        super(Resnet18Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


