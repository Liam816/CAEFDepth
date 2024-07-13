import os
import argparse
from utils.main_utils import parse_arguments, load_config_file
from utils import logger
from model import loader
import multiprocessing
import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from torch import Tensor
from torchvision import transforms
from typing import Tuple


def normalization2img(x: np.ndarray, mode='linear'):
    """
    :param x:
    :param mode: 'linear'--let x in [0, 255], 'std'--standard deviation
    :return:
    """
    if mode == 'linear':
        vmax = np.max(x)
        vmin = np.min(x)
        res = (x - vmin) / (vmax - vmin)
        res = res * 255
        return res.astype(np.uint8)
    elif mode == 'std':
        mu = np.sum(x) / x.size
        deviation = np.sqrt((x - mu) / x.size)
        res = (x - mu) / deviation
        return res


def edge_salience_score(edge_img, indices: Tuple, crop=None, mode="horizontal"):
    """
    :param edge_img: 网络预测出的单通道的深度图（没有经过裁剪）
    :param indices: 经过裁剪后的图像中的edge像素点位置
    :param crop: 对图像边缘的裁剪，左右上下四个边界，List
    :return:
    """
    if crop is not None:
        edge_img_croped = edge_img[crop[0]:crop[1], crop[2]:crop[3], :]
    else:
        edge_img_croped = edge_img

    h, w, _ = edge_img_croped.shape

    y_loc = indices[0]
    x_loc = indices[1]
    indices_up = (y_loc - 1, x_loc)
    indices_down = (y_loc + 1, x_loc)
    indices_left = (y_loc, x_loc - 1)
    indices_right = (y_loc, x_loc + 1)

    # 防止超出图像边界
    indices_up[0][indices_up[0] > (h - 1)] = h - 1
    indices_up[0][indices_up[0] < 0] = 0
    indices_up[1][indices_up[1] > (w - 1)] = w - 1
    indices_up[1][indices_up[1] < 0] = 0

    indices_down[0][indices_down[0] > (h - 1)] = h - 1
    indices_down[0][indices_down[0] < 0] = 0
    indices_down[1][indices_down[1] > (w - 1)] = w - 1
    indices_down[1][indices_down[1] < 0] = 0

    indices_left[0][indices_left[0] > (h - 1)] = h - 1
    indices_left[0][indices_left[0] < 0] = 0
    indices_left[1][indices_left[1] > (w - 1)] = w - 1
    indices_left[1][indices_left[1] < 0] = 0

    indices_right[0][indices_right[0] > (h - 1)] = h - 1
    indices_right[0][indices_right[0] < 0] = 0
    indices_right[1][indices_right[1] > (w - 1)] = w - 1
    indices_right[1][indices_right[1] < 0] = 0

    assert indices[0].size == indices[1].size, "Sorry!"
    edge_pixel_num = indices[0].size

    sum_res_hv = 0
    for i in range(edge_pixel_num):
        center = edge_img_croped[indices[0][i], indices[1][i], :]
        up = edge_img_croped[indices_up[0][i], indices_up[1][i], :]
        down = edge_img_croped[indices_down[0][i], indices_down[1][i], :]
        left = edge_img_croped[indices_left[0][i], indices_left[1][i], :]
        right = edge_img_croped[indices_right[0][i], indices_right[1][i], :]

        temp = np.abs(center - up) + np.abs(center - down) + np.abs(center - left) + np.abs(center - right)
        temp = temp / (255 * 4)
        sum_res_hv = sum_res_hv + temp

    sum_res_h = 0
    for i in range(edge_pixel_num):
        center = edge_img_croped[indices[0][i], indices[1][i], :]
        left = edge_img_croped[indices_left[0][i], indices_left[1][i], :]
        right = edge_img_croped[indices_right[0][i], indices_right[1][i], :]
        temp = np.abs(center - left) + np.abs(center - right)
        temp = temp / (255 * 2)
        sum_res_h = sum_res_h + temp

    sum_res_v = 0
    for i in range(edge_pixel_num):
        center = edge_img_croped[indices[0][i], indices[1][i], :]
        up = edge_img_croped[indices_up[0][i], indices_up[1][i], :]
        down = edge_img_croped[indices_down[0][i], indices_down[1][i], :]
        temp = np.abs(center - up) + np.abs(center - down)
        temp = temp / (255 * 2)
        sum_res_v = sum_res_v + temp

    # sum_res = 0
    # if mode == "horizontal":
    #     for i in range(edge_pixel_num):
    #         center = edge_img_croped[indices[0][i], indices[1][i], :]
    #         left = edge_img_croped[indices_left[0][i], indices_left[1][i], :]
    #         right = edge_img_croped[indices_right[0][i], indices_right[1][i], :]
    #         temp = np.abs(center - left) + np.abs(center - right)
    #         temp = temp / (255 * 2)
    #         sum_res = sum_res + temp
    # elif mode == "vertical":
    #     for i in range(edge_pixel_num):
    #         center = edge_img_croped[indices[0][i], indices[1][i], :]
    #         up = edge_img_croped[indices_up[0][i], indices_up[1][i], :]
    #         down = edge_img_croped[indices_down[0][i], indices_down[1][i], :]
    #         temp = np.abs(center - up) + np.abs(center - down)
    #         temp = temp / (255 * 2)
    #         sum_res = sum_res + temp

    return sum_res_h, sum_res_v, sum_res_hv


# def main():
#     opts = parse_arguments()
#     opts2 = parse_arguments()
#     # config_file_path = "./config/MDE_MobileViTv1-s_GUB.yaml"
#     # config_file_path = "./config/MDE_MobileViTv2-1.0_GUB.yaml"
#     # config_file_path = "./config/MDE_MobileViTv1-s_GUB_Bin.yaml"
#     # config_file_path = "./config/MDE_MobileViTv1-s_GUB_AdaBins.yaml"
#     # config_file_path = "./config/MDE_MobileViTv3-s_GUB.yaml"
#     # config_file_path = "./config/MDE_FasterNet-T2_GUB.yaml"
#     # config_file_path = "./config/MDE_DDRNet-23-slim_GUB.yaml"
#     # config_file_path = "./config/MDE_FasterNet-S_GUB.yaml"
#     # config_file_path = "./config/MDE_FasterNet-S_PCGUB.yaml"
#     # config_file_path = "./config/MDE_FasterNet-S_EdgePCGUB.yaml"
#     # config_file_path = "./config/MDE_MobileNetV2_EGN.yaml"
#     config_file_path = "./config/MDE_DDRNet-23-slim_LiamEdge.yaml"
#     # config_file_path = "./config/MDE_MobileNetV2_LiamEdge.yaml"
#     # NOTE：在yaml文件中配置mode为'eval'，在loader.load_model()中修改选择对应的chechpoint_x.pth
#     opts = load_config_file(config_file_path, opts)
#
#     config_file_path = "./config/MDE_FasterNet-S_PCGUB.yaml"
#     opts2 = load_config_file(config_file_path, opts2)
#
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     model = loader.load_model(opts)
#     logger.info("model.__class__.__name__:{}".format(model.__class__.__name__))
#     model.to(device=device)
#     model.eval()
#
#     model2 = loader.load_model(opts2)
#     logger.info("model2.__class__.__name__:{}".format(model2.__class__.__name__))
#     model2.to(device=device)
#     model2.eval()
#
#     cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
#     cmap_type = cmap_type_list[2]
#
#     """ 获取边缘像素索引 """
#     crop = [20, 460, 24, 616]
#     img = cv.imread('/home/glw/hp/datasets/nyu_data/data/nyu2_test/00578_depth.png', -1)
#     img = normalization2img(img)
#     img = np.expand_dims(img, axis=2)
#     img_ = img  # 未裁剪
#     img = img[crop[0]:crop[1], crop[2]:crop[3]]  # 剪裁边缘用以计算edge indices (mask)
#     img = cv.applyColorMap(img, cv.COLORMAP_PLASMA)
#     canny_img = cv.Canny(img, 100, 200)  # threshold:64, 128  shape:[h, w]
#     canny_img = np.expand_dims(canny_img, axis=2)  # [h, w]->[h, w, 1]
#     indices = np.where(canny_img > 200)
#     """ 获取边缘像素索引 """
#
#     canny_img_ = cv.Canny(img_, 100, 200)  # 未裁剪
#     canny_img_ = np.expand_dims(canny_img_, axis=2)
#     gt_canny_res_h, gt_canny_res_v, gt_canny_res_hv = edge_salience_score(canny_img_, indices)
#     gt_depth_res_h, gt_depth_res_v, gt_depth_res_hv = edge_salience_score(img_, indices)
#     # gt_canny_res_v = edge_salience_score(canny_img_, indices, mode="vertical")
#     # gt_depth_res_v = edge_salience_score(img_, indices, mode="vertical")
#     print("gt_canny_res horizontal:{} vertical:{} hv:{}".format(gt_canny_res_h, gt_canny_res_v, gt_canny_res_hv))
#     print("gt_depth_res horizontal:{} vertical:{} hv:{}".format(gt_depth_res_h, gt_depth_res_v, gt_depth_res_hv))
#
#     """ 模型推理得到估计的单通道深度图 """
#     transformation = transforms.ToTensor()
#
#     # depth = cv.imread("/home/glw/hp/datasets/nyu_data/data/nyu2_test/00578_depth.png", -1)
#     # depth = depth * 0.001  # millimeter -> meter
#     # depth = depth.astype(np.float32)
#     # depth = transformation(depth)
#     # depth = depth.unsqueeze(0).to(device=device)
#     # print("type(depth):", type(depth))
#     # print("depth.shape:", depth.shape)
#
#     image = cv.imread("/home/glw/hp/datasets/nyu_data/data/nyu2_test/00578_colors.png", -1)
#     image = image.astype(np.float32) / 255.0
#     image = transformation(image)
#     image = image.unsqueeze(0).to(device=device)
#     # print("type(image):", type(image))
#     # print("image.shape:", image.shape)
#
#     res = model(image)
#     res = res.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     # print("res.shape:", res.shape)
#     # print("max:{} min:{}".format(np.max(res), np.min(res)))
#     res = normalization2img(res)
#     # print("res.shape:", res.shape)
#
#     res2 = model2(image)
#     res2 = res2.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
#     res2 = normalization2img(res2)
#     """ 模型推理得到估计的单通道深度图 """
#
#     estimated_depth_res_h, estimated_depth_res_v, estimated_depth_res_hv = edge_salience_score(res, indices)
#     # estimated_depth_res_v = edge_salience_score(res, indices, mode="vertical")
#     print("estimated_depth_res horizontal:{} vertical:{} hv:{}"
#           .format(estimated_depth_res_h, estimated_depth_res_v, estimated_depth_res_hv))
#     estimated_depth_res2_h, estimated_depth_res2_v, estimated_depth_res2_hv = edge_salience_score(res2, indices)
#     # estimated_depth_res2_v = edge_salience_score(res2, indices, mode="vertical")
#     print("estimated_depth_res2 horizontal:{} vertical:{} hv:{}"
#           .format(estimated_depth_res2_h, estimated_depth_res2_v, estimated_depth_res2_hv))
#
#
#     """ 可视化 """
#     fig = plt.figure(figsize=(6, 1))  # 宽度 高度
#     ax1 = fig.add_axes([0.02, 0.1, 0.33, 0.95])  # 离左边界距离 离下边界距离 宽度比例 高度比例
#     ax1.imshow(img_, cmap_type_list[0])
#     ax2 = fig.add_axes([0.35, 0.1, 0.33, 0.95])
#     ax2.imshow(res, cmap_type_list[0])  # canny_img
#     ax3 = fig.add_axes([0.67, 0.1, 0.33, 0.95])
#     ax3.imshow(res2, cmap_type_list[0])
#     plt.show()
#
#     # plt.imshow(canny_img, cmap_type_list[0])
#     # plt.imshow(img_, cmap_type_list[0])
#     # plt.show()


def main():
    """
    对DecoderBlockV5(CBAM)add操作前后的特征图做评估
    """
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
    config_file_path = "./config/MDE_DDRNet-slim_LiamEdge.yaml"
    # config_file_path = "./config/MDE_MobileNetV2_LiamEdge.yaml"
    # NOTE：在yaml文件中配置mode为'eval'，在loader.load_model()中修改选择对应的chechpoint_x.pth
    opts = load_config_file(config_file_path, opts)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = loader.load_model(opts)
    logger.info("model.__class__.__name__:{}".format(model.__class__.__name__))
    model.to(device=device)
    model.eval()

    cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
    cmap_type = cmap_type_list[2]

    """ 获取边缘像素索引 """
    crop = [20, 460, 24, 616]
    img = cv.imread('/home/glw/hp/datasets/nyu_data/data/nyu2_test/00578_depth.png', -1)
    img = normalization2img(img)
    img = np.expand_dims(img, axis=2)
    img_ = img  # 未裁剪
    img = img[crop[0]:crop[1], crop[2]:crop[3]]  # 剪裁边缘用以计算edge indices (mask)
    img_r2 = cv.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    img_r4 = cv.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
    img_r8 = cv.resize(img, (0, 0), fx=0.125, fy=0.125, interpolation=cv.INTER_LINEAR)
    # print("r2 h:{} w:{}".format(img_r2.shape[0], img_r2.shape[1]))
    # print("r4 h:{} w:{}".format(img_r4.shape[0], img_r4.shape[1]))
    # print("r8 h:{} w:{}".format(img_r8.shape[0], img_r8.shape[1]))

    img = cv.applyColorMap(img, cv.COLORMAP_PLASMA)
    canny_img = cv.Canny(img, 100, 200)  # threshold:64, 128  shape:[h, w]
    canny_img = np.expand_dims(canny_img, axis=2)  # [h, w]->[h, w, 1]
    indices = np.where(canny_img > 200)

    img_r2 = cv.applyColorMap(img_r2, cv.COLORMAP_PLASMA)
    canny_img_r2 = cv.Canny(img_r2, 100, 200)  # threshold:64, 128  shape:[h, w]
    canny_img_r2 = np.expand_dims(canny_img_r2, axis=2)  # [h, w]->[h, w, 1]
    indices_r2 = np.where(canny_img_r2 > 200)

    img_r4 = cv.applyColorMap(img_r4, cv.COLORMAP_PLASMA)
    canny_img_r4 = cv.Canny(img_r4, 100, 200)  # threshold:64, 128  shape:[h, w]
    canny_img_r4 = np.expand_dims(canny_img_r4, axis=2)  # [h, w]->[h, w, 1]
    indices_r4 = np.where(canny_img_r4 > 200)

    img_r8 = cv.applyColorMap(img_r8, cv.COLORMAP_PLASMA)
    canny_img_r8 = cv.Canny(img_r8, 100, 200)  # threshold:64, 128  shape:[h, w]
    canny_img_r8 = np.expand_dims(canny_img_r8, axis=2)  # [h, w]->[h, w, 1]
    indices_r8 = np.where(canny_img_r8 > 200)

    # fig = plt.figure(figsize=(6, 2))  # 宽度 高度
    # ax1 = fig.add_axes([0.02, 0.52, 0.3, 0.45])  # 离左边界距离 离下边界距离 宽度比例 高度比例
    # ax1.imshow(img_r2, cmap_type_list[0])
    # ax2 = fig.add_axes([0.35, 0.52, 0.3, 0.45])
    # ax2.imshow(img_r4, cmap_type_list[0])  # canny_img
    # ax3 = fig.add_axes([0.67, 0.52, 0.3, 0.45])
    # ax3.imshow(img_r8, cmap_type_list[0])
    # ax4 = fig.add_axes([0.02, 0.02, 0.3, 0.45])  # 离左边界距离 离下边界距离 宽度比例 高度比例
    # ax4.imshow(canny_img_r2, cmap_type_list[0])
    # ax5 = fig.add_axes([0.35, 0.02, 0.3, 0.45])
    # ax5.imshow(canny_img_r4, cmap_type_list[0])  # canny_img
    # ax6 = fig.add_axes([0.67, 0.02, 0.3, 0.45])
    # ax6.imshow(canny_img_r8, cmap_type_list[0])
    # plt.show()
    # exit()
    """ 获取边缘像素索引 """

    canny_img_ = cv.Canny(img_, 100, 200)  # 未裁剪
    canny_img_ = np.expand_dims(canny_img_, axis=2)
    gt_canny_res_h, gt_canny_res_v, gt_canny_res_hv = edge_salience_score(canny_img_, indices, crop=[20, 460, 24, 616])
    gt_depth_res_h, gt_depth_res_v, gt_depth_res_hv = edge_salience_score(img_, indices, crop=[20, 460, 24, 616])
    # gt_canny_res_v = edge_salience_score(canny_img_, indices, mode="vertical")
    # gt_depth_res_v = edge_salience_score(img_, indices, mode="vertical")
    print("gt_canny_res horizontal:{} vertical:{} hv:{}".format(gt_canny_res_h, gt_canny_res_v, gt_canny_res_hv))
    print("gt_depth_res horizontal:{} vertical:{} hv:{}".format(gt_depth_res_h, gt_depth_res_v, gt_depth_res_hv))

    """ 模型推理得到估计的单通道深度图 """
    transformation = transforms.ToTensor()
    image = cv.imread("/home/glw/hp/datasets/nyu_data/data/nyu2_test/00578_colors.png", -1)
    print("image.shape: ", image.shape)
    image = image[crop[0]:crop[1], crop[2]:crop[3], :]
    print("image.shape: ", image.shape)
    image = image.astype(np.float32) / 255.0
    image = transformation(image)
    image = image.unsqueeze(0).to(device=device)
    # print("type(image):", type(image))
    print("image.shape:", image.shape)
    res, temp1, temp2 = model(image)
    res = res.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    temp1 = temp1.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    temp2 = temp2.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    print("res.shape:", res.shape)
    print("max:{} min:{}".format(np.max(res), np.min(res)))
    print("temp1.shape:", temp1.shape)
    print("temp2.shape:", temp2.shape)
    res = normalization2img(res)
    temp1 = normalization2img(temp1)
    temp2 = normalization2img(temp2)
    """ 模型推理得到估计的单通道深度图 """

    # flag = 5
    # temp1 = temp1[:, :, flag:flag + 1]
    # temp2 = temp2[:, :, flag:flag + 1]
    #
    # estimated_depth_res_h, estimated_depth_res_v, estimated_depth_res_hv \
    #     = edge_salience_score(res, indices, crop=[20, 460, 24, 616])
    # temp1_h, temp1_v, temp1_hv = edge_salience_score(temp1, indices_r2)
    # temp2_h, temp2_v, temp2_hv = edge_salience_score(temp2, indices_r2)
    # print("estimated_depth_res horizontal:{} vertical:{} hv:{}"
    #       .format(estimated_depth_res_h, estimated_depth_res_v, estimated_depth_res_hv))
    # print("temp1_before_add horizontal:{} vertical:{} hv:{}".format(temp1_h, temp1_v, temp1_hv))
    # print("temp2_after_add horizontal:{} vertical:{} hv:{}".format(temp2_h, temp2_v, temp2_hv))
    #
    # """ 可视化 """
    # fig = plt.figure(figsize=(12, 3))  # 宽度 高度
    # ax1 = fig.add_axes([0.03, 0.06, 0.3, 0.95])  # 离左边界距离 离下边界距离 宽度比例 高度比例
    # ax1.imshow(img_r2, cmap_type_list[0])
    # ax2 = fig.add_axes([0.36, 0.06, 0.3, 0.95])
    # ax2.imshow(temp1, cmap_type_list[0])  # canny_img
    # ax3 = fig.add_axes([0.69, 0.06, 0.3, 0.95])
    # ax3.imshow(temp2, cmap_type_list[0])
    # plt.show()

    for i in range(6, 16, 1):
        x = temp1[:, :, i:i+1]
        y = temp2[:, :, i:i+1]

        estimated_depth_res_h, estimated_depth_res_v, estimated_depth_res_hv \
            = edge_salience_score(res, indices, crop=[20, 460, 24, 616])
        temp1_h, temp1_v, temp1_hv = edge_salience_score(x, indices_r2)
        temp2_h, temp2_v, temp2_hv = edge_salience_score(y, indices_r2)
        print("estimated_depth_res horizontal:{} vertical:{} hv:{}"
              .format(estimated_depth_res_h, estimated_depth_res_v, estimated_depth_res_hv))
        print("temp1_before_add horizontal:{} vertical:{} hv:{}".format(temp1_h, temp1_v, temp1_hv))
        print("temp2_after_add horizontal:{} vertical:{} hv:{}".format(temp2_h, temp2_v, temp2_hv))

        """ 可视化 """
        fig = plt.figure(figsize=(12, 3))  # 宽度 高度
        ax1 = fig.add_axes([0.03, 0.06, 0.3, 0.95])  # 离左边界距离 离下边界距离 宽度比例 高度比例
        ax1.imshow(img_r2, cmap_type_list[0])
        ax2 = fig.add_axes([0.36, 0.06, 0.3, 0.95])
        ax2.imshow(x, cmap_type_list[0])  # canny_img
        ax3 = fig.add_axes([0.69, 0.06, 0.3, 0.95])
        ax3.imshow(y, cmap_type_list[0])
        plt.show()

    # plt.imshow(canny_img, cmap_type_list[0])
    # plt.imshow(img_, cmap_type_list[0])
    # plt.show()


if __name__ == "__main__":
    main()
