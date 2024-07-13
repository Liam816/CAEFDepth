import os.path

import torch.nn as nn
import torch

from model.GuideDepth import GuideDepth
from nets.models.MobileViT import MobileViT
from nets.models.MobileViT_GUB import MobileViTGUB
from nets.models.MobileViTv2_GUB import MobileViTv2GUB
from nets.models.MobileViTv3_GUB import MobileViTv3GUB
from nets.models.MobileViT_GUB_Bin import MobileViTGUBBIN
from nets.models.MobileViT_GUB_AdaBins import MobileViTGUBAdaBins
from nets.models.FasterNet_GUB import FasterNetGUB
from nets.models.FasterNet_PCGUB import FasterNetPCGUB
from nets.models.FasterNet_EdgePCGUB import FasterNetEdgePCGUB
from nets.models.DDRNet_LiamEdge import DDRNetEdge
from nets.models.MobileNetV2_EGN import MobileNetV2EGN
from nets.models.MobileNetV2_LiamEdge import MobileNetV2Edge
from nets.models.MobileViTv1_LiamEdge import MobileViTv1Edge
from nets.models.FasterNet_LiamEdge import FasterNetEdge
# from nets.models.VanillaNet import VanillaNet
from nets.models.MobileOne import mobileone
from nets.models.DDRNet_LiamEdge_V2 import DDRNetEdgeV2
from nets.models.FastViT_LiamEdge import FastViTEdge
from nets.models.FastDepth import FastDepthMobileNet, FastDepthMobileNetSkipAdd, FastDepthMobileNetSkipConcat
from utils import logger


# def load_model(model_name, weights_pth, opts):
#     model = model_builder(model_name)
#     # if weights_pth is not None:
#     #     state_dict = torch.load(weights_pth, map_location='cpu')
#     #     model.load_state_dict(state_dict)
#     if weights_pth is not None:
#         weights = torch.load(weights_pth, map_location='cpu')
#         weights_dict = {}
#         for k, v in weights.items():
#             new_k = k.replace('module.', '') if 'module' in k else k
#             weights_dict[new_k] = v
#         model.load_state_dict(weights_dict)
#
#     return model

# def model_builder(model_name):
#     if model_name == 'GuideDepth':
#         return GuideDepth(True)
#     if model_name == 'GuideDepth-S':
#         return GuideDepth(True, up_features=[32, 8, 4], inner_features=[32, 8, 4])
#
#     print("Invalid model")
#     exit(0)


# def load_model(opts):
#     mode = getattr(opts, "common.mode", "train")
#     model_name = getattr(opts, "model.name", "GuideDepth")
#
#     logger.log("mode:{}".format(mode))
#     logger.log("model_name:{}".format(model_name))
#
#     if mode == "train":
#         if model_name == 'GuideDepth':
#             model = GuideDepth(True)
#         elif model_name == 'GuideDepth-S':
#             model = GuideDepth(True, up_features=[32, 8, 4], inner_features=[32, 8, 4])
#         elif model_name == 'MobileViT':
#             model = MobileViT(opts)
#         elif model_name == "MobileViTGUB":
#             model = MobileViTGUB(opts)
#         elif model_name == "MobileViTv2GUB":
#             model = MobileViTv2GUB(opts)
#         elif model_name == "MobileViTGUBBin":
#             model = MobileViTGUBBIN(opts)
#         elif model_name == "MobileViTGUBAdaBins":
#             model = MobileViTGUBAdaBins(opts)
#         elif model_name == "MobileViTv3GUB":
#             model = MobileViTv3GUB(opts)
#         elif model_name == "FasterNetGUB":
#             model = FasterNetGUB(opts)
#         elif model_name == 'DDRNetGUB':
#             model = GuideDepth(True)
#         elif model_name == "FasterNetPCGUB":
#             model = FasterNetPCGUB(opts)
#         elif model_name == "FasterNetEdgePCGUB":
#             model = FasterNetEdgePCGUB(opts)
#         elif model_name == "DDRNetEdge":
#             model = DDRNetEdge(opts)
#         else:
#             logger.error("{} is not supported yet.".format(model_name))
#
#         pretrained_path = getattr(opts, "model.pretrained", None)
#         if pretrained_path is not None:
#             weights = torch.load(pretrained_path)
#             # logger.log("weights.device:{}".format(weights.device))
#             weights_dict = {}
#             for k, v in weights.items():
#                 new_k = k.replace('module.', '') if 'module' in k else k
#                 weights_dict[new_k] = v
#
#             # MiT模型删除classifier部分权重
#             del_keys = []
#             for k, v in weights_dict.items():
#                 if "classifier" in k:
#                     del_keys.append(k)
#             for i in range(len(del_keys)):
#                 weights_dict.pop(del_keys[i])
#
#             # print("weights_dict.keys():\n", weights_dict.keys(), sep="\n")
#             missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
#             if len(unexpected_keys) > 0:
#                 logger.log("There exists unexpected keys.")
#                 print("[unexpected_keys]:", *unexpected_keys, sep="\n")
#             if len(missing_keys) > 0:
#                 logger.log("There exists missing keys.")
#                 # print("[missing_keys]:", *missing_keys, sep="\n")
#             logger.info("Loaded pretrained weights: {}".format(pretrained_path))
#
#     elif mode == "eval":
#         if model_name == 'GuideDepth':
#             model = GuideDepth(True)
#         elif model_name == 'GuideDepth-S':
#             model = GuideDepth(True, up_features=[32, 8, 4], inner_features=[32, 8, 4])
#         elif model_name == 'MobileViT':
#             model = MobileViT(opts)
#         elif model_name == "MobileViTGUB":
#             model = MobileViTGUB(opts)
#         elif model_name == "MobileViTv2GUB":
#             model = MobileViTv2GUB(opts)
#         elif model_name == "MobileViTGUBBin":
#             model = MobileViTGUBBIN(opts)
#         elif model_name == "MobileViTGUBAdaBins":
#             model = MobileViTGUBAdaBins(opts)
#         elif model_name == "MobileViTv3GUB":
#             model = MobileViTv3GUB(opts)
#         elif model_name == "FasterNetGUB":
#             model = FasterNetGUB(opts)
#         elif model_name == 'DDRNetGUB':
#             model = GuideDepth(True)
#         elif model_name == "FasterNetPCGUB":
#             model = FasterNetPCGUB(opts)
#         elif model_name == "FasterNetEdgePCGUB":
#             model = FasterNetEdgePCGUB(opts)
#         elif model_name == "DDRNetEdge":
#             model = DDRNetEdge(opts)
#         else:
#             logger.error("{} is not supported yet.".format(model_name))
#
#         # print("model.state_dict():\n", model.state_dict().keys())
#         # exit()
#
#         checkpoint_path = getattr(opts, "common.save_checkpoint", None)
#         # best_model_path = os.path.join(checkpoint_path, "best_model", "best_model.pth")
#         best_model_path = os.path.join(checkpoint_path, "checkpoint_32.pth")
#         # logger.log("best_model_path:{}".format(best_model_path))
#
#         weights = torch.load(best_model_path)
#         weights = weights["model"]
#         weights_dict = {}
#         for k, v in weights.items():
#             new_k = k.replace('module.', '') if 'module' in k else k
#             weights_dict[new_k] = v
#
#         missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
#         if len(unexpected_keys) > 0:
#             logger.log("There exists unexpected keys.")
#             print("[unexpected_keys]:", *unexpected_keys, sep="\n")
#         if len(missing_keys) > 0:
#             logger.log("There exists missing keys.")
#             print("[missing_keys]:", *missing_keys, sep="\n")
#
#     return model

def load_model(opts):
    mode = getattr(opts, "common.mode", "train")
    model_name = getattr(opts, "model.name", "GuideDepth")
    model_mode = getattr(opts, 'model.mode')

    # logger.log("mode:{}".format(mode))
    logger.log("model_name:{}".format(model_name))

    if model_name == "GuideDepth":
        model = GuideDepth(opts, True)
    elif model_name == "GuideDepth-S":
        model = GuideDepth(opts, True, up_features=[32, 8, 4], inner_features=[32, 8, 4])
    elif model_name == "MobileViT":
        model = MobileViT(opts)
    elif model_name == "MobileViTGUB":
        model = MobileViTGUB(opts)
    elif model_name == "MobileViTv2GUB":
        model = MobileViTv2GUB(opts)
    elif model_name == "MobileViTGUBBin":
        model = MobileViTGUBBIN(opts)
    elif model_name == "MobileViTGUBAdaBins":
        model = MobileViTGUBAdaBins(opts)
    elif model_name == "MobileViTv3GUB":
        model = MobileViTv3GUB(opts)
    elif model_name == "FasterNetGUB":
        model = FasterNetGUB(opts)
    elif model_name == "DDRNetGUB":
        model = GuideDepth(opts, True)
    elif model_name == "FasterNetPCGUB":
        model = FasterNetPCGUB(opts)
    elif model_name == "FasterNetEdgePCGUB":
        model = FasterNetEdgePCGUB(opts)
    elif model_name == "DDRNetEdge":
        model = DDRNetEdge(opts)
    elif model_name == "MobileNetV2EGN":
        model = MobileNetV2EGN(opts)
    elif model_name == "MobileNetV2Edge":
        model = MobileNetV2Edge(opts)
    elif model_name == "MobileViTv1Edge":
        model = MobileViTv1Edge(opts)
    elif model_name == "FasterNetEdge":
        model = FasterNetEdge(opts)
    elif model_name == "MobileOne":
        model = mobileone(opts)
    elif model_name == 'DDRNetEdgeV2':
        model = DDRNetEdgeV2(opts)
    elif model_name == 'FastViTEdge':
        model = FastViTEdge(opts)
    elif model_name == 'FastDepth':
        if model_mode == 'MN':
            model = FastDepthMobileNet()
        elif model_mode == 'MNSkipAdd':
            model = FastDepthMobileNetSkipAdd()
        elif model_mode == 'MNSkipConcat':
            model = FastDepthMobileNetSkipConcat()
    else:
        logger.error("{} is not supported yet.".format(model_name))

    if mode == "train":
        pretrained_path = getattr(opts, "model.pretrained", None)
        logger.log("pretrained_path:{}".format(pretrained_path))
        if pretrained_path is not None:
            weights = torch.load(pretrained_path)  # 默认只包含model 也可能加载的是包含各种信息的checkpoint

            if "model" in weights.keys():
                weights = weights["model"]

            # logger.log("weights.device:{}".format(weights.device))
            weights_dict = {}
            for k, v in weights.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v

            # # MiT模型删除classifier部分权重
            # del_keys = []
            # for k, v in weights_dict.items():
            #     if "classifier" in k:
            #         del_keys.append(k)
            # for i in range(len(del_keys)):
            #     weights_dict.pop(del_keys[i])

            # MiT模型删除classifier部分权重
            if 'MobileViT' in model_name:
                del_keys = []
                for k, v in weights_dict.items():
                    if "classifier" in k:
                        del_keys.append(k)
                for i in range(len(del_keys)):
                    weights_dict.pop(del_keys[i])

            if model_name == 'MobileNetV2Edge' and model_mode == 'TokenPyramid':
                weights_dict = weights_dict['state_dict']
                # 删除除了tpm(TokenPyramidModule)以外的其他权重
                del_keys = []
                for k, v in weights_dict.items():
                    if "tpm" not in k:
                        del_keys.append(k)
                for i in range(len(del_keys)):
                    weights_dict.pop(del_keys[i])
                # print('weights_dict.keys():{}'.format(weights_dict.keys()))

                weights_dict_new = {}
                for k, v in weights_dict.items():
                    new_k = k.replace('backbone.tpm', 'rgb_features_extractor')
                    weights_dict_new[new_k] = v
                weights_dict = weights_dict_new
                # print('weights_dict_new.keys():{}'.format(weights_dict_new.keys()))

            missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)

            if len(unexpected_keys) > 0:
                logger.log("There exists unexpected keys.")
                # print("[unexpected_keys]:", *unexpected_keys, sep="\n")
            if len(missing_keys) > 0:
                logger.log("There exists missing keys.")
                # print("[missing_keys]:", *missing_keys, sep="\n")

            if len(unexpected_keys) == 0 and len(missing_keys) == 0:
                logger.info("Loaded pretrained weights w/o unexpected or missing keys.\nweights path: {}".format(pretrained_path))

    elif mode == "eval":
        checkpoint_path = getattr(opts, "common.save_checkpoint", None)
        best_checkpoint = getattr(opts, "common.best_checkpoint", None)
        best_model_path = os.path.join(checkpoint_path, best_checkpoint)

        # NOTE: 需要测试官方的模型权重文件 直接指定路径
        # best_model_path = '/home/data/glw/hp/models/GuidedDecoding/KITTI_Full_GuideDepth.pth'
        # best_model_path = './checkpoints/MobileNetV2-TP_ENV4_TMV3M4_LearnableAlpha_OHV4_240628_1/checkpoint_19.pth'
        logger.log("best_model_path:{}".format(best_model_path))

        weights = torch.load(best_model_path)
        weights = weights["model"]  # NOTE: 测试GuidedDecoding的kitti模型需要注释掉该行
        # print('=== weight keys ===')
        # for k, v in weights.items():
        #     print(k)
        # print('=== weight keys ===')
        # exit()

        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        # NOTE: MobileNetV2_LiamEdge.py中的MobileNetV2Edge模型需要做以下操作
        if getattr(opts, "model.name", None) == 'MobileNetV2Edge':
            # weights_dict = {}
            # for k, v in weights.items():
            #     # new_k = k.replace('module.', '') if 'module.' in k and 'transition_module.' not in k else k
            #     new_k = k[7:]
            #     weights_dict[new_k] = v

            # print('=== weight keys ===')
            # for k, v in weights_dict.items():
            #     print(k)
            # print('=== weight keys ===')
            # exit()
            
            weights_dict_new = {}
            for k, v in weights_dict.items():
                new_k = k.replace('extension_extension_', 'extension_module.extension_') if 'extension_extension_module_IRB' in k else k
                weights_dict_new[new_k] = v
            weights_dict = weights_dict_new
            # print("weights.keys():{}".format(weights.keys()))

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(unexpected_keys) > 0:
            logger.log("There exists unexpected keys.")
            print("[unexpected_keys]:", *unexpected_keys, sep="\n")
        if len(missing_keys) > 0:
            logger.log("There exists missing keys.")
            print("[missing_keys]:", *missing_keys, sep="\n")

        logger.log('The best model weights loaded w/o unexpected or missing keys.')

    return model

