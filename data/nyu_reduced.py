from venv import logger
import pandas as pd
import numpy as np
import torch
import torchvision
import os
from zipfile import ZipFile
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from data.MyTransforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor

import cv2 as cv
import matplotlib.pyplot as plt
import csv


resolution_dict = {
    'full': (480, 640),
    'half': (240, 320),
    'mini': (224, 224)}


def data_normalization(x):
    vmin = np.min(x)
    vmax = np.max(x)
    return (((x-vmin)/(vmax-vmin))*255).astype(np.uint8)


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


class depthDatasetMemory(Dataset):
    def __init__(self, data, split, nyu2_train, transform=None, train_edge=False, use_depthnorm=True):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform
        self.split = split
        self.train_edge = train_edge

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        # print('sample:{}'.format(sample))
        # exit()
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))
        image = np.array(image).astype(np.float32)  # [h, w, 3]
        depth = np.array(depth).astype(np.float32)  # [h, w]

        # print('image.shape:{}'.format(image.shape))
        # print('depth.shape:{}'.format(depth.shape))

        # NOTE: Generating gt depth edge on the fly
        if self.train_edge:  # 训练集的深度图本身就是[0, 255]图像像素范围的
            if self.split == 'train':
                # h, w, c = image.shape
                # h_, w_ = depth.shape
                # assert h == h_ and h == 480 and w == w_ and w == 640, print('The shape of image and depth are wrong.')
                # crop = [20, 460, 24, 616]
                # depth_ = depth.astype(np.uint8)
                # depth_ = cv.resize(depth_, (640, 480))  # (width, height)
                # depth_ = depth_[crop[0]:crop[1], crop[2]:crop[3]]  # 裁剪边缘
                # depth_ = cv.applyColorMap(depth_, cv.COLORMAP_PLASMA)  # 灰度图转彩图
                # depth_edge = cv.Canny(depth_, 100, 200)
                # depth_edge = np.expand_dims(depth_edge, axis=2)  # [h, w]->[h, w, 1]
                # # 裁剪之后的边缘索引 不能直接return出去 因为torch的dataloader需要对齐batch_size个样本 每个样本的数据形状都得一致
                # # edge_indices = np.where(depth_canny > 200)
                # transformation = transforms.ToTensor()
                # depth_edge = transformation(depth_edge)  # numpy [h, w, 1] -> tensor [1, h, w]

                # fig, axes = plt.subplots(1, 3)
                # # 在第一个子图中显示image1
                # axes[0].imshow(image.astype(np.uint8))
                # axes[0].set_title('image')
                # axes[1].imshow(depth.astype(np.uint8), cmap='gray')  # plasma
                # axes[1].set_title('depth')
                # axes[2].imshow(depth_canny, cmap='gray')
                # axes[2].set_title('depth_canny')
                # plt.tight_layout()  # 调整子图之间的间距
                # plt.show()
                # exit()

                temp = sample[0].split('/')
                scene = temp[-2]
                sn = temp[-1].split('.')[0]
                root = '/home/data/glw/hp/datasets/nyu/nyu2_train_edge_gt/'
                depth_edge = Image.open(os.path.join(root, scene, sn + '_edge.png'))
                depth_edge = np.array(depth_edge).astype(np.uint8)  # [h, w]
                # print('depth_edge.shape:{}'.format(depth_edge.shape))
                transformation = transforms.ToTensor()
                depth_edge = transformation(depth_edge)  # numpy [h, w] -> tensor [1, h, w]

                # plt.imshow(depth_edge)
                # plt.show()
                # exit()
            elif self.split == 'val':
                temp = sample[0].split('/')
                sn = temp[-1].split('_')[0]
                root = '/home/data/glw/hp/datasets/nyu/nyu2_test_edge_gt/'
                depth_edge = Image.open(os.path.join(root, sn + '_depth_edge.png'))
                depth_edge = np.array(depth_edge).astype(np.uint8)  # [h, w]
                # print('depth_edge.shape:{}'.format(depth_edge.shape))
                transformation = transforms.ToTensor()
                depth_edge = transformation(depth_edge)  # numpy [h, w] -> tensor [1, h, w]

        if self.split == 'train':
            depth = depth / 255.0 * 10.0  # From 8bit to range [0, 10] (meter)
        elif self.split == 'val':
            depth = depth * 0.001  # millimeter -> meter

        # print('image.shape:{} depth.shape:{}'.format(image.shape, depth.shape))
        # print('image|max:{} min:{}\ndepth|max:{} min:{}'.format(np.max(image), np.min(image),
        #                                                         np.max(depth), np.min(depth)))
        # exit()

        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        if self.train_edge:
            sample['depth_edge'] = depth_edge

        return sample

    def __len__(self):
        return len(self.nyu_dataset)


class depthDatasetExtracted(Dataset):
    def __init__(self, dataset_list, split, transform=None, train_edge=False, use_depthnorm=True):
        self.nyu_dataset = dataset_list
        self.transform = transform
        self.split = split
        self.train_edge = train_edge

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        # print('sample:{}'.format(sample))
        # exit()
        image = Image.open(sample[0])
        depth = Image.open(sample[1]).convert('L') 
        image = np.array(image).astype(np.float32)  # [h, w, 3]
        depth = np.array(depth).astype(np.float32)  # [h, w]
        
        # print('image.shape:', image.shape)
        # print('image max:{} min:{}'.format(np.max(image), np.min(image)))
        # print('depth.shape:', depth.shape)
        # print('depth max:{} min:{}'.format(np.max(depth), np.min(depth)))
        # exit()

        if self.train_edge:  # 训练集的深度图本身就是[0, 255]图像像素范围的
            if self.split == 'train':
                temp = sample[0].split('/')
                scene = temp[-2]
                sn = temp[-1].split('.')[0]
                root = '/home/data/glw/hp/datasets/nyu/nyu2_train_edge_gt/'
                depth_edge = Image.open(os.path.join(root, scene, sn + '_edge.png'))
                depth_edge = np.array(depth_edge).astype(np.uint8)  # [h, w]
                transformation = transforms.ToTensor()
                depth_edge = transformation(depth_edge)  # numpy [h, w] -> tensor [1, h, w]
            elif self.split == 'val':
                temp = sample[0].split('/')
                sn = temp[-1].split('_')[0]
                root = '/home/data/glw/hp/datasets/nyu/nyu2_test_edge_gt/'
                depth_edge = Image.open(os.path.join(root, sn + '_depth_edge.png'))
                depth_edge = np.array(depth_edge).astype(np.uint8)  # [h, w]
                transformation = transforms.ToTensor()
                depth_edge = transformation(depth_edge)  # numpy [h, w] -> tensor [1, h, w]

        if self.split == 'train':
            depth = depth / 255.0 * 10.0  # From 8bit to range [0, 10] (meter)
        elif self.split == 'val':
            depth = depth * 0.001  # millimeter -> meter

        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)

        if self.train_edge:
            sample['depth_edge'] = depth_edge

        return sample

    def __len__(self):
        return len(self.nyu_dataset)


# class NYU_Testset_Extracted(Dataset):
#     def __init__(self, root, resolution='full'):
#         self.root = root
#         self.resolution = resolution_dict[resolution]
#
#         self.files = os.listdir(self.root)
#
#     def __getitem__(self, index):
#         image_path = os.path.join(self.root, self.files[index])
#
#         data = np.load(image_path)
#         depth, image = data['depth'], data['image']
#         depth = np.expand_dims(depth, axis=2)
#
#         image, depth = data['image'], data['depth']
#         image = np.array(image)
#         depth = np.array(depth)
#         return image, depth
#
#     def __len__(self):
#         return len(self.files)


class NYU_Testset_Extracted(Dataset):
    def __init__(self, root):
        # print('flag1')
        rgb_np = np.load(os.path.join(root, 'eigen_test_rgb.npy'))
        # print('flag2')
        depth_np = np.load(os.path.join(root, 'eigen_test_depth.npy'))
        # print('flag3')
        self.rgb = torch.from_numpy(rgb_np).type(torch.float32)  # Range [0,1]
        # print('flag4')
        print('self.rgb shape:', self.rgb.shape)
        # print('self.rgb[0] shape:', self.rgb[0].shape)
        # print('self.rgb[0] max:{} min:{}'.format(torch.max(self.rgb[0]), torch.min(self.rgb[0])))
        self.depth = torch.from_numpy(depth_np).type(torch.float32)  # Range [0, 10]
        print('self.depth shape:', self.depth.shape)
        # print('self.depth[0] max:{} min:{}'.format(torch.max(self.depth[0]), torch.min(self.depth[0])))
        # print('flag5')
        # exit()

    def __getitem__(self, idx):
        image = self.rgb[idx]
        depth = self.depth[idx]
        return image, depth

    def __len__(self):
        return len(self.rgb)


class NYU_Testset(Dataset):
    def __init__(self, zip_path):
        input_zip = ZipFile(zip_path)
        data = {name: input_zip.read(name) for name in input_zip.namelist()}

        self.rgb = torch.from_numpy(np.load(BytesIO(data['eigen_test_rgb.npy']))).type(torch.float32)  # Range [0,1]
        self.depth = torch.from_numpy(np.load(BytesIO(data['eigen_test_depth.npy']))).type(torch.float32)  # Range [0, 10]

    def __getitem__(self, idx):
        image = self.rgb[idx]
        depth = self.depth[idx]
        # print("in class NYU_Testset image max:{}, min:{}".format(torch.max(image), torch.min(image)))
        # print("in class NYU_Testset depth max:{}, min:{}".format(np.max(depth), np.min(depth)))
        return image, depth

    def __len__(self):
        return len(self.rgb)


def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    input_zip = ZipFile(zip_file)
    print('flag1')
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    print('flag2')
    nyu2_train = list(
        (row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list(
        (row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))
    print('flag3')

    # Debugging
    # if True: nyu2_train = nyu2_train[:100]
    # if True: nyu2_test = nyu2_test[:100]

    print('Loaded (Train Images: {0}, Test Images: {1}).'.format(len(nyu2_train), len(nyu2_test)))
    return data, nyu2_train, nyu2_test


def getDatasetsExtracted(root, split):
    if split == 'train':
        # csv_file = os.path.join(root, 'data', 'nyu2_train.csv')
        # csv_file = os.path.join(root, 'data', 'nyu2_train_new.csv')
        csv_file = os.path.join(root, 'data', 'nyu2_train_raw.csv')
    elif split == 'val':
        csv_file = os.path.join(root, 'data', 'nyu2_test.csv')

    dataset = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append((os.path.join(root, row[0]), os.path.join(root, row[1])))

    return dataset


def train_transform(resolution):
    transform = transforms.Compose([
        Resize(resolution),
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor(test=False, maxDepth=10.0)
    ])
    return transform


def val_transform(resolution):
    transform = transforms.Compose([
        Resize(resolution),
        ToTensor(test=True, maxDepth=10.0)
    ])
    return transform


# NOTE: 原版
# def get_NYU_dataset(zip_path, split, resolution='full', model_name="GuideDepth", uncompressed=False, train_edge=False, use_depthnorm=True):
#     resolution = resolution_dict[resolution]
#     if split == 'train':
#         data, nyu2_train, nyu2_test = loadZipToMem(zip_path)
#
#         transform = train_transform(resolution)
#         dataset = depthDatasetMemory(data, split, nyu2_train, transform=transform, train_edge=train_edge)
#     elif split == 'val':
#         data, nyu2_train, nyu2_test = loadZipToMem(zip_path)
#
#         transform = val_transform(resolution)
#         dataset = depthDatasetMemory(data, split, nyu2_test, transform=transform, train_edge=train_edge)
#     elif split == 'test':
#         if uncompressed:
#             dataset = NYU_Testset_Extracted(zip_path)
#         else:
#             dataset = NYU_Testset(zip_path)
#
#     return dataset


def get_NYU_dataset(root, split, resolution='full', uncompressed=False, train_edge=False, use_depthnorm=True):
    resolution = resolution_dict[resolution]

    print('LIAM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    if split != 'test':  # 训练集和验证集
        if split == 'train':
            transform = train_transform(resolution)
        elif split == 'val':
            transform = val_transform(resolution)

        dataset_list = getDatasetsExtracted(root, split)
        # logger.info('found {} samples in {} set'.format(len(dataset_list), split))
        print('found {} samples in {} set'.format(len(dataset_list), split))
        dataset = depthDatasetExtracted(dataset_list, split, transform=transform, train_edge=train_edge)

    else:  # 测试集
        if uncompressed:
            dataset = NYU_Testset_Extracted(root)
        else:
            dataset = NYU_Testset(root)

    return dataset


