import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DDRNet_23_slim import DualResNet_Backbone
from model.modules import Guided_Upsampling_Block, SELayer
import time


class GuideDepth(nn.Module):
    def __init__(self,
                 opts,
                 pretrained=True,
                 up_features=[64, 32, 16],
                 inner_features=[64, 32, 16]):
        super(GuideDepth, self).__init__()

        self.feature_extractor = DualResNet_Backbone(
            opts,
            pretrained=pretrained,
            features=up_features[0]
        )

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                            expand_features=inner_features[0],
                                            out_features=up_features[1],
                                            kernel_size=3,
                                            channel_attention=True,
                                            guide_features=3,
                                            guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                            expand_features=inner_features[1],
                                            out_features=up_features[2],
                                            kernel_size=3,
                                            channel_attention=True,
                                            guide_features=3,
                                            guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                            expand_features=inner_features[2],
                                            out_features=1,
                                            kernel_size=3,
                                            channel_attention=True,
                                            guide_features=3,
                                            guidance_type="full")

    def forward(self, x):
        t0 = time.time()
        y = self.feature_extractor(x)  # [b, 64, 60, 80]
        # print("DDRNet time:{:.4f}s".format(time.time()-t0))
        y = y[-1]  # LIAM

        t0 = time.time()
        x_half = F.interpolate(x, scale_factor=.5)
        x_quarter = F.interpolate(x, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')  # [b, 64, 120, 160]
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')  # [b, 64, 240, 320]
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2.0, mode='bilinear')  # [b, 64, 480, 640]
        y = self.up_3(x, y)
        # print("3 layers GUB time:{:.4f}s".format(time.time() - t0))

        return y
