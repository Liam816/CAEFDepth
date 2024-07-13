import os
import argparse

from training import Trainer
# from training_for_test import Trainer
from evaluate import Evaluater
# from evaluate_for_test import Evaluater
from utils.main_utils import parse_arguments, load_config_file
from utils import logger
from model import loader
from data import datasets
from data.bts_dataloader import BtsDataLoader
import multiprocessing
import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn.functional as F



def normalize(x: Tensor):
    min_val = x.min()
    max_val = x.max()
    return (x - min_val) / (max_val - min_val)


def normalize2img(x: Tensor):
    min_val = x.min()
    max_val = x.max()
    res = ((x - min_val) / (max_val - min_val)) * 255
    res = res.detach().cpu().numpy().astype(np.uint8)
    return res


def edge_extractor(x, mode, device='cpu'):
    b, c, h, w = x.size()
    x_ = x
    x_ = x_ * 255
    x_ = x_.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)  # [b, h, w, c]

    if mode == 'sobel':
        # NOTE: Sobel
        edge_batch_tensor = torch.randn(size=(b, h, w, 3))
        for i in range(b):
            Sobelx = cv.Sobel(x_[i, :, :, :], cv.CV_8U, 1, 0)  # 输出unit8类型的图像
            Sobely = cv.Sobel(x_[i, :, :, :], cv.CV_8U, 0, 1)
            Sobelx = cv.convertScaleAbs(Sobelx)
            Sobely = cv.convertScaleAbs(Sobely)
            Sobelxy = cv.addWeighted(Sobelx, 0.5, Sobely, 0.5, 0)  # [h, w, 3]
            # Sobelxy = Sobelxy.transpose(2, 0, 1)  # [3, h, w]
            edge_batch_tensor[i, :, :, :] = torch.from_numpy(Sobelxy).type(torch.float32)
        edge = edge_batch_tensor.to(device)  # [b, h, w, 3]
    elif mode == 'canny':
        # NOTE: Canny
        edge_batch_tensor = torch.randn(size=(b, h, w, 1))
        for i in range(b):
            canny_edge = cv.Canny(x[i, :, :, :], 100, 200)
            canny_edge = np.expand_dims(canny_edge, axis=2)  # [h, w, 1]
            canny_edge = normalize(canny_edge)  # 将数据缩放到[0, 1]区间
            edge_batch_tensor[i, :, :, :] = torch.from_numpy(canny_edge).type(torch.float32)
        edge = edge_batch_tensor.to(device)  # [b, 1, h, w]
    elif mode == 'laplacian':
        # NOTE: Laplacian TODO
        pass

    return edge


def main():
    # torch.manual_seed(6)
    # torch.cuda.manual_seed_all(6)

    opts = parse_arguments()
    # config_file_path = "./config/MDE_MobileViTv1-s_GUB.yaml"
    # config_file_path = "./config/MDE_MobileViTv2-1.0_GUB.yaml"
    # config_file_path = "./config/MDE_MobileViTv1-s_GUB_Bin.yaml"
    # config_file_path = "./config/MDE_MobileViTv1-s_GUB_AdaBins.yaml"
    # config_file_path = "./config/MDE_MobileViTv3-s_GUB.yaml"
    # config_file_path = "./config/MDE_FasterNet-T2_GUB.yaml"
    # config_file_path = "./config/MDE_DDRNet-23-slim_GUB.yaml"
    # config_file_path = "./config/MDE_FasterNet-S_GUB.yaml"
    # config_file_path = "./config/MDE_FasterNet-S_PCGUB.yaml"
    # config_file_path = "./config/MDE_FasterNet-S_EdgePCGUB.yaml"
    # config_file_path = "./config/MDE_MobileNetV2_EGN.yaml"
    # config_file_path = "./config/MDE_DDRNet-23-slim_LiamEdge.yaml"
    config_file_path = './config/MDE_MobileNetV2_LiamEdge.yaml'
    # config_file_path = './config/MDE_FasterNet-X_LiamEdge.yaml'

    # NOTE：在yaml文件中配置mode为'eval'，在loader.load_model()中修改选择对应的chechpoint_x.pth
    opts = load_config_file(config_file_path, opts)

    taskname = getattr(opts, "taskname", None)
    logger.log('taskname: {}'.format(taskname))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    model = loader.load_model(opts)
    logger.info("model.__class__.__name__:{}".format(model.__class__.__name__))
    model.to(device)
    model.eval()

    dataset_name = getattr(opts, "dataset.name", "nyu_reduced")
    model_name = getattr(opts, "model.name", "GuideDepth")
    data_path = getattr(opts, "dataset.root", None)
    test_path = getattr(opts, "dataset.root_test", None)
    eval_mode = getattr(opts, "common.eval_mode", "alhashim")
    batch_size = getattr(opts, "common.bs", 8)
    resolution = getattr(opts, "common.resolution", "full")
    train_edge = getattr(opts, "common.train_edge", False)
    n_cpus = multiprocessing.cpu_count()
    num_workers = n_cpus // 2

    # dataset_name = 'kitti'
    # data_path = '/home/data/glw/hp/datasets/kitti_eigen/test_dataset.zip'
    # data_path = '/home/data/glw/hp/datasets/kitti_eigen/train_dataset'

    logger.info("dataset_name:{}".format(dataset_name))
    logger.info("model_name:{}".format(model_name))
    logger.info("data_path:{}".format(data_path))
    logger.info("test_path:{}".format(test_path))
    logger.info("eval_mode:{}".format(eval_mode))
    logger.info("batch_size:{}".format(batch_size))
    logger.info("resolution:{}".format(resolution))
    logger.info("num_workers:{}".format(num_workers))
    logger.info("train_edge:{}".format(train_edge))

    # train_loader = datasets.get_dataloader(dataset_name,
    #                                        model_name,
    #                                        path=data_path,
    #                                        split='train',
    #                                        augmentation=eval_mode,
    #                                        batch_size=batch_size,
    #                                        resolution=resolution,
    #                                        workers=num_workers,
    #                                        train_edge=train_edge)

    # val_loader = datasets.get_dataloader(dataset_name,
    #                                      model_name,
    #                                      path=data_path,
    #                                      split='val',
    #                                      augmentation=eval_mode,
    #                                      batch_size=batch_size,
    #                                      resolution=resolution,
    #                                      workers=num_workers)

    # test_loader = datasets.get_dataloader(dataset_name,
    #                                       model_name,
    #                                       path=test_path,
    #                                       split='test',
    #                                       batch_size=1,
    #                                       augmentation=eval_mode,
    #                                       resolution='full',  # resolution_opt
    #                                       workers=num_workers)

    # train_loader = BtsDataLoader(opts, 'train')
    val_loader = BtsDataLoader(opts, 'online_eval')

    # for i, data in enumerate(train_loader.data):
    #     if i % 1000 == 0:
    #         print('i:{}'.format(i))
    # print('train dataset inspected.')

    for i, data in enumerate(val_loader.data):
        if i % 100 == 0:
            print('i:{}'.format(i))
    print('test dataset inspected.')


    # data = next(iter(train_loader.data))
    exit()

    # NOTE: 加载数据集中的图片
    # print('len(test_loader):{}'.format(len(test_loader)))
    data = next(iter(val_loader))
    image = data['image'].to(device, non_blocking=True)
    # depth = data['depth']
    print('image.shape:{}'.format(image.shape))

    _, attn_score_list, x_list = model(image)

    H, W = 480, 640
    image32 = F.interpolate(image, scale_factor=0.03125, mode='bilinear', align_corners=False)
    image16 = F.interpolate(image, scale_factor=0.0625, mode='bilinear', align_corners=False)
    image8 = F.interpolate(image, scale_factor=0.125, mode='bilinear', align_corners=False)

    temp = image[0, :, :, :]  # [1, 3, h, w]
    print('temp.shape:{}'.format(temp.shape))
    temp = normalize2img(temp)
    temp = temp.transpose(1, 2, 0)  # [h, w, 3]

    temp32 = image32[0, :, :, :]  # [1, 3, h, w]
    temp32 = normalize2img(temp32)
    temp32 = temp32.transpose(1, 2, 0)  # [h, w, 3]

    temp16 = image16[0, :, :, :]  # [1, 3, h, w]
    temp16 = normalize2img(temp16)
    temp16 = temp16.transpose(1, 2, 0)  # [h, w, 3]

    temp8 = image8[0, :, :, :]  # [1, 3, h, w]
    temp8 = normalize2img(temp8)
    temp8 = temp8.transpose(1, 2, 0)  # [h, w, 3]

    for i in range(len(attn_score_list)):
        print('attn_score_list[{}].shape:{}'.format(i, attn_score_list[i].shape))

    for i in range(len(x_list)):
        print('x_list[{}].shape:{}'.format(i, x_list[i].shape))

    fm_list = []
    # 放大到对应特征图尺寸
    for i, x in enumerate(x_list):
        if i < 3:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        elif i < 6:
            x = F.interpolate(x, scale_factor=4.0, mode='bilinear', align_corners=False)
        elif i < 9:
            x = F.interpolate(x, scale_factor=8.0, mode='bilinear', align_corners=False)
        fm_list.append(x)

    for i in range(len(fm_list)):
        print('fm_list[{}].shape:{}'.format(i, fm_list[i].shape))

    fm = fm_list[-1][0, :, :, :]
    print('fm.shape:{}'.format(fm.shape))
    fm = normalize2img(fm)
    fm = fm.transpose(1, 2, 0)  # [h, w, 3]

    cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
    fig, axes = plt.subplots(1, 2)
    # 在第一个子图中显示image1
    axes[0].imshow(temp8)
    axes[0].set_title('image')
    # 在第二个子图中显示image2
    axes[1].imshow(fm[:, :, 0:1], cmap='plasma')
    axes[1].set_title('fm')
    # axes[1].imshow(x_edge[0], cmap='plasma')
    # axes[1].set_title('Image 2')
    plt.tight_layout()  # 调整子图之间的间距
    plt.show()

    exit()





    data = next(iter(train_loader))
    image = data['image']
    print('image.shape:{}'.format(image.shape))
    temp1 = image[0, 0:1, :, :]
    temp2 = image[0, 1:2, :, :]
    temp3 = image[0, 2:, :, :]
    print('temp1 max:{} min:{}'.format(temp1.max(), temp1.min()))
    print('temp2 max:{} min:{}'.format(temp2.max(), temp2.min()))
    print('temp3 max:{} min:{}'.format(temp3.max(), temp3.min()))

    # data = next(iter(val_loader))
    exit()

    data = next(iter(train_loader))
    image = data['image']
    depth = data['depth']
    print('image.shape:{} depth.shape:{}'.format(image.shape, depth.shape))
    print('image|max:{} min:{}\ndepth|max:{} min:{}'.format(torch.max(image), torch.min(image),
                                                            torch.max(depth), torch.min(depth)))
    exit()

    image, depth = next(iter(test_loader))
    print('image.shape:{} depth:{}'.format(image.shape, depth.shape))

    for i in range(image.shape[0]):
        print('image max:{} min:{}'.format(torch.max(image[i]), torch.min(image[i])))
        print('depth max:{} min:{}'.format(torch.max(depth[i]), torch.min(depth[i])))
    exit()

    image, depth = next(iter(test_loader))

    for i in range(image.shape[0]):
        # print('type(image):{}'.format(type(image)))
        # print('image.shape:{}'.format(image.shape))
        print('image max:{} min:{}'.format(torch.max(image[i]), torch.min(image[i])))
        # print('type(depth):{}'.format(type(depth)))
        # print('depth.shape:{}'.format(depth.shape))
        print('depth max:{} min:{}'.format(torch.max(depth[i]), torch.min(depth[i])))
    exit()

    # print('image batch[0]:\n', image[0])
    # print('image batch[0] max:{} min:{}'.format(image[0].max(), image[0].min()))

    # NOTE: 可视化边缘图
    # X = torch.randn(size=(4, 3, 480, 640)).to(device)
    # _, edges, before_add, after_add = model(image)
    _, edges = model(image)
    temp = edges[1][0, :, :, :]  # [c, h, w]
    print("temp.shape:", temp.shape)
    print("temp max:{} min:{}".format(temp.max(), temp.min()))
    temp = normalize2img(temp)
    temp = temp.transpose(1, 2, 0)
    print("temp.shape:", temp.shape)

    raw_edge = edge_extractor(image, 'sobel')
    raw_edge = raw_edge.numpy().astype(np.uint8)

    image = image * 255
    image = image.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

    cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
    cmap_type = cmap_type_list[2]

    fig = plt.figure(figsize=(12, 9))
    for i in range(16):
        if i < 4:
            ax = fig.add_axes([0.02 + 0.245 * i, 0.75, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
            ax.imshow(temp[:, :, i:i+1], cmap_type_list[0])
        elif i < 8:
            ax = fig.add_axes([0.02 + 0.245 * (i - 4), 0.50, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
            ax.imshow(temp[:, :, i:i+1], cmap_type_list[0])
        elif i < 12:
            ax = fig.add_axes([0.02 + 0.245 * (i - 8), 0.25, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
            ax.imshow(temp[:, :, i:i+1], cmap_type_list[0])
        else:
            ax = fig.add_axes([0.02 + 0.245 * (i - 12), 0.00, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
            ax.imshow(temp[:, :, i:i+1], cmap_type_list[0])
    plt.show()

    # ===========================================================================

    # NOTE: 可视化注意力图
    _, temp1, temp2 = model(image)
    for i in range(len(temp1)):
        print('temp1[{}].shape:{}'.format(i, temp1[i].shape))
        print('max:{} min:{}'.format(temp1[i].max(), temp1[i].min()))
    for i in range(len(temp2)):
        print('temp2[{}].shape:{}'.format(i, temp2[i].shape))
        print('max:{} min:{}'.format(temp2[i].max(), temp2[i].min()))

    # # [c, h, w]  某一张图片-》网络输出的多个通道图
    # temp1 = temp1[0][0, :, :, :]  # before attn
    # temp2 = temp2[0][0, :, :, :]  # after attn
    # # exit()
    # temp1 = normalize2img(temp1)
    # temp1 = temp1.transpose(1, 2, 0)
    # temp2 = normalize2img(temp2)
    # temp2 = temp2.transpose(1, 2, 0)
    #
    # cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
    # cmap_type = cmap_type_list[2]
    #
    # fig = plt.figure(figsize=(12, 9))
    # for i in range(16):
    #     if i < 4:
    #         ax = fig.add_axes([0.02 + 0.245 * i, 0.75, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
    #         ax.imshow(temp1[:, :, i:i+1], cmap_type_list[0])
    #     elif i < 8:
    #         ax = fig.add_axes([0.02 + 0.245 * (i - 4), 0.50, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
    #         ax.imshow(temp2[:, :, i:i+1], cmap_type_list[0])
    #     elif i < 12:
    #         ax = fig.add_axes([0.02 + 0.245 * (i - 8), 0.25, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
    #         ax.imshow(temp1[:, :, i:i+1], cmap_type_list[0])
    #     else:
    #         ax = fig.add_axes([0.02 + 0.245 * (i - 12), 0.00, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
    #         ax.imshow(temp2[:, :, i:i+1], cmap_type_list[0])
    # plt.show()

    image = image * 255
    image = image.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)  # [b, h, w, c]
    print('image.shape:{}'.format(image.shape))

    for j in range(batch_size):
        fig_ = plt.figure()
        ax_ = fig_.add_axes([0.05, 0.05, 0.9, 0.9])
        ax_.imshow(image[j])
        # plt.imshow()
        # plt.show()

        # [c, h, w]  batch中的一张图片-》网络输出的多个通道图
        x = temp1[0][j, :, :, :]  # before attn
        y = temp2[0][j, :, :, :]  # after attn
        # exit()
        x = normalize2img(x)
        x = x.transpose(1, 2, 0)
        y = normalize2img(y)
        y = y.transpose(1, 2, 0)

        cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
        cmap_type = cmap_type_list[2]

        fig = plt.figure(figsize=(12, 9))
        for i in range(16):
            if i < 4:
                ax = fig.add_axes([0.02 + 0.245 * i, 0.75, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
                ax.imshow(x[:, :, i:i + 1], cmap_type_list[0])
            elif i < 8:
                ax = fig.add_axes([0.02 + 0.245 * (i - 4), 0.50, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
                ax.imshow(y[:, :, i:i + 1], cmap_type_list[0])
            elif i < 12:
                ax = fig.add_axes([0.02 + 0.245 * (i - 8), 0.25, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
                ax.imshow(x[:, :, i:i + 1], cmap_type_list[0])
            else:
                ax = fig.add_axes([0.02 + 0.245 * (i - 12), 0.00, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
                ax.imshow(y[:, :, i:i + 1], cmap_type_list[0])
        plt.show()

    # for i in range(8):
    #     fig = plt.figure(figsize=(12, 3))
    #     ax1 = fig.add_axes([0.03, 0.06, 0.3, 0.95])  # 离左边界距离 离下边界距离 宽度比例 高度比例
    #     ax1.imshow(raw_edge[0], cmap_type_list[0])
    #     ax2 = fig.add_axes([0.36, 0.06, 0.3, 0.95])
    #     ax2.imshow(temp[:, :, i:i+1], cmap_type_list[0])  # canny_img
    #     ax3 = fig.add_axes([0.69, 0.06, 0.3, 0.95])
    #     ax3.imshow(temp[:, :, i+1:i+2], cmap_type_list[0])
    #     plt.show()

    # for i in range(6):
    #     temp1 = normalize2img(before_add[0][0, i*8:i*8+1, ::])
    #     temp1 = temp1.transpose(1, 2, 0)
    #     temp2 = normalize2img(after_add[0][0, i*8:i*8+1, ::])
    #     temp2 = temp2.transpose(1, 2, 0)
    #     # ax1 = fig.add_axes([0.02+0.16*i, 0.55, 0.2, 0.2])
    #     # ax1.imshow(temp1)
    #     # ax2 = fig.add_axes([0.02+0.16*i, 0.05, 0.2, 0.2])
    #     # ax2.imshow(temp2)
    #     plt.subplot(2, 6, i+1)
    #     plt.imshow(temp1, cmap_type)
    #     plt.subplot(2, 6, i+7)
    #     plt.imshow(temp2, cmap_type)
    # plt.show()

    # plt.subplot(3, 4, 1)
    # plt.imshow(image[0])
    #
    # plt.subplot(3, 4, 2)
    # plt.imshow(raw_edge[0])
    # plt.subplot(2, 3, 4)
    # plt.imshow(raw_edge[0][:, :, :1], cmap_type)
    # plt.subplot(2, 3, 5)
    # plt.imshow(raw_edge[0][:, :, 1:2], cmap_type)
    # plt.subplot(2, 3, 6)
    # plt.imshow(raw_edge[0][:, :, 2:3], cmap_type)

    # [B, 16, 240, 320]  !
    # [B, 64, 120, 160]  !
    # [B, 64, 60, 80]    !
    # edges[0][0, :1, ::]  # [1, 240, 320]
    # temp = normalize2img(edges[0][0, :1, ::])
    # temp = temp.transpose(1, 2, 0)
    # print('edges[0][0, :1, ::]', edges[0][0, :1, ::])
    # print('edges[0][0, :1, ::] max:{} min:{}'.format(edges[0][0, :1, ::].max(), edges[0][0, :1, ::].min()))
    #
    # plt.subplot(2, 3, 3)
    # plt.imshow(temp, cmap_type)
    # plt.show()

    # for i in range(8):
    #     temp = normalize2img(edges[0][0, i:i+1, ::])
    #     temp = temp.transpose(1, 2, 0)
    #     plt.subplot(3, 4, i+5)
    #     plt.imshow(temp, cmap_type)

    # plt.show()

    # NOTE: Visualize raw images
    # cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
    # cmap_type = cmap_type_list[1]
    # image = image * 255
    # image = image.detach().to(device).numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    # plt.imshow(image[0])
    # plt.show()

    # NOTe: 原始边缘
    # raw_edge = edge_extractor(image, 'sobel')
    # raw_edge = raw_edge.numpy().astype(np.uint8)
    # for i in range(batch_size):
    #     plt.subplot(2, batch_size//2, i + 1)
    #     plt.imshow(raw_edge[i])
    # plt.show()


if __name__ == "__main__":
    main()
