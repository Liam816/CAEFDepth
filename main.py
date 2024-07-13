import os
import argparse

# from training import Trainer
# from training_latest import Trainer

# from evaluate import Evaluater
# from evaluate_latest import Evaluater

from utils.main_utils import parse_arguments, load_config_file
from utils import logger


def main():
    args = parse_arguments()
    # config_file_path = "./config/MDE_MobileViTv1-s_GUB.yaml"
    # config_file_path = "./config/MDE_MobileViTv2-1.0_GUB.yaml"
    # config_file_path = "./config/MDE_MobileViTv1-s_GUB_Bin.yaml"
    # config_file_path = "./config/MDE_MobileViTv1-s_GUB_AdaBins.yaml"
    # config_file_path = "./config/MDE_MobileViTv3-s_GUB.yaml"
    # config_file_path = "./config/MDE_FasterNet-T2_GUB.yaml"
    # config_file_path = "./config/MDE_FasterNet-S_GUB.yaml"
    # config_file_path = "./config/MDE_FasterNet-S_PCGUB.yaml"
    # config_file_path = "./config/MDE_FasterNet-S_EdgePCGUB.yaml"
    # config_file_path = "./config/MDE_MobileNetV2_EGN.yaml"

    # config_file_path = './config/MDE_VanillaNet_LiamEdge.yaml'
    # config_file_path = './config/MDE_MobileOne_LiamEdge.yaml'
    # config_file_path = 'config/MDE_DDRNet-trim_LiamEdge_V2.yaml'
    # config_file_path = './config/MDE_MobileViTv1-s_LiamEdge.yaml'

    # config_file_path = './config/MDE_DDRNet-23-slim_LiamEdge.yaml'
    # config_file_path = 'config/MDE_DDRNet-trim_LiamEdge_V2.yaml'

    config_file_path = './config/MDE_DDRNet-23-slim_GUB.yaml'
    # config_file_path = './config/MDE_MobileNetV2_LiamEdge.yaml'

    # config_file_path = './config/MDE_FastViT_LiamEdge.yaml'
    # config_file_path = './config/MDE_FasterNet-X_LiamEdge.yaml'
    # config_file_path = 'config/MDE_FastDepth.yaml'

    opts = load_config_file(config_file_path, args)

    dataset = getattr(opts, 'dataset.name', 'nyu_reduced')
    mode = getattr(opts, 'common.mode', 'train')
    # print('dataset:{}'.format(dataset))
    # print('mode:{}'.format(mode))

    # if mode == 'train':
    #     model_trainer = Trainer(opts)
    #     model_trainer.train(opts)
    # elif mode == 'eval':
    #     evaluation_module = Evaluater(opts)
    #     evaluation_module.evaluate()

    if mode == 'train':
        if dataset == 'nyu_reduced':
            from training import Trainer
        elif dataset == 'kitti':
            # from training_latest import Trainer
            from training import Trainer
        model_trainer = Trainer(opts)
        model_trainer.train(opts)
    elif mode == 'eval':
        if dataset == 'nyu_reduced':
            from evaluate import Evaluater
        elif dataset == 'kitti':
            # from evaluate_latest import Evaluater
            from evaluate import Evaluater
        evaluation_module = Evaluater(opts)
        evaluation_module.evaluate()
    else:
        logger.error('Mode \"{}\" is not supported yet.'.format(mode))


if __name__ == '__main__':
    main()
