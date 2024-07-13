import numpy as np
import torch
import os
from zipfile import ZipFile
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from data.MyTransforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor, CenterCrop, RandomRotation, \
    RandomVerticalFlip
# from MyTransforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor, CenterCrop, RandomRotation, \
#     RandomVerticalFlip

import time
import random
import csv


resolution_dict = {
    'full': (384, 1280),
    # 'full': (352, 1216),  # bts训练时先裁剪了边缘
    'tu_small': (128, 416),
    'tu_big': (228, 912),
    'half': (192, 640)}


class KITTIDataset(Dataset):
    def __init__(self, root, data, data_indices, split, resolution='full', augmentation='alhashim'):
        # self.root = root
        # self.split = split
        # self.resolution = resolution_dict[resolution]
        # self.augmentation = augmentation
        #
        # if split == 'train':
        #     self.transform = self.train_transform
        #     self.root = os.path.join(self.root, 'train')
        # elif split == 'val':
        #     self.transform = self.val_transform
        #     self.root = os.path.join(self.root, 'val')
        # elif split == 'test':
        #     if self.augmentation == 'alhashim':
        #         self.transform = None
        #     else:
        #         self.transform = CenterCrop(self.resolution)
        #
        #     self.root = os.path.join(self.root, 'test')
        #
        # self.files = os.listdir(self.root)  # .npz文件 numpy格式的数据

        # NOTE: LIAM
        self.root = root
        self.split = split
        self.data = data
        self.data_indices = data_indices
        self.resolution = resolution_dict[resolution]
        self.augmentation = augmentation

        if split == 'train':
            self.transform = self.train_transform
        elif split == 'val':
            self.transform = self.val_transform
        elif split == 'test':
            if self.augmentation == 'alhashim':
                self.transform = None
            else:
                self.transform = CenterCrop(self.resolution)

    def __getitem__(self, index):
        # image_path = os.path.join(self.root, self.files[index])
        #
        # data = np.load(image_path)
        # depth, image = data['depth'], data['image']
        #
        # if self.transform is not None:
        #     data = self.transform(data)
        #
        # image, depth = data['image'], data['depth']
        # if self.split == 'test':
        #     image = np.array(image)
        #     depth = np.array(depth)
        # return image, depth

        # NOTE: LIAM
        sample = self.data_indices[index]
        # print('sample:{} {}'.format(sample[0], sample[1]))
        # zip_file_name = self.root.split('/')[-1]
        # root = self.root.replace()
        # sample[0] = sample[0][1:]
        # sample[1] = sample[1][1:]
        # print('self.data.get(sample[0]):{}'.format(self.data.get(sample[0])))
        sample[0] = 'test_dataset' + sample[0]
        sample[1] = 'test_dataset' + sample[1]
        # print('sample[0]:{}'.format(sample[0]))
        # print('sample[1]:{}'.format(sample[1]))
        # print('====================')
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = np.load(BytesIO(self.data[sample[1]]), allow_pickle=True)  # 已经是meter单位的深度数据
        image = np.array(image).astype(np.float32)  # RGB图像原本就是[0, 255]之间的，改成浮点型数据

        if self.split == 'test':
            image = torch.from_numpy(image).type(torch.float32)  # [0.0, 255.0]
            depth = torch.from_numpy(depth).type(torch.float32)  # [0.0, 80.0]

        sample_ = {'image': image, 'depth': depth}  # train val

        if self.transform is not None:
            sample_ = self.transform(sample_)  # train val image归一化到[0.0 1.0]之间

        return sample_

    def __len__(self):
        return len(self.data_indices)

    def train_transform(self, data):  # train训练集
        if self.augmentation == 'alhashim':
            transform = Compose([
                Resize(self.resolution),
                RandomHorizontalFlip(),
                RandomChannelSwap(0.25),
                ToTensor(test=False, maxDepth=80.0)
            ])
        else:
            transform = Compose([
                RandomRotation(4.5),
                CenterCrop(self.resolution),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomChannelSwap(0.25),
                ToTensor(test=False, maxDepth=80.0)
            ])

        data = transform(data)
        return data

    def val_transform(self, data):  # validation验证集
        if self.augmentation == 'alhashim':
            transform = Compose([
                Resize(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])
        else:
            transform = Compose([
                CenterCrop(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])

        data = transform(data)
        return data


class KITTIDataset_(Dataset):
    def __init__(self, root, data, data_indices, split, resolution='full', augmentation='alhashim'):
        """
        Args:
            root:
            data: 如果是train和val,则为纯粹的数据路径构成的列表;如果是test,则为zipfile读取的数据路径构成的字典
            data_indices:
            split:
            resolution:
            augmentation:
        """
        # NOTE: LIAM
        self.root = root
        self.split = split
        self.data = data
        self.data_indices = data_indices
        self.resolution = resolution_dict[resolution]
        self.augmentation = augmentation

        # print('augmentation in class {} is : {}'.format(self.__class__, augmentation))

        if split == 'train':
            self.transform = self.train_transform
        elif split == 'val':
            self.transform = self.val_transform
            # self.transform = self.test_transform  # NOTE: 验证集充当测试集
            # self.transform = None
        elif split == 'test':
            if self.augmentation == 'alhashim':
                self.transform = None  # TODO: 是否需要将RGB图像从[0, 255]映射到[0.0 1.0]之间
                # self.transform = self.test_transform
                # self.transform = self.val_transform  # NOTE: 测试集充当验证集
            else:
                self.transform = CenterCrop(self.resolution)

        # 使用稀疏的gt来做eval 用稠密化后的本身就引入了更多的误差
        if split == 'test':
            filenames_file_eval = './train_test_inputs/eigen_test_files_with_gt.txt'
            with open(filenames_file_eval, 'r') as f:  # 697个样本 但其中有部分gt深度图为None的情况 将这部分数据已经去除
                self.filenames = f.readlines()

    def __getitem__(self, index):
        # NOTE: 适用于现在划分train和val
        # if self.split == 'train' or self.split == 'val':
        #     sample = self.data[self.data_indices[index]]  # 包含image和depth两个路径的列表 并且其路径开头有一个'/'需要去掉
        #     image_path = os.path.join(self.root, sample[0][1:])
        #     depth_path = os.path.join(self.root, sample[1][1:])
        #
        #     image = Image.open(image_path)
        #     image = np.array(image).astype(np.float32)  # RGB图像原本就是[0, 255]之间的，改成浮点型数据
        #     depth = np.load(depth_path, allow_pickle=True)
        #     # image = torch.from_numpy(image).type(torch.float32)  # [0.0, 255.0]
        #
        #     sample_ = {'image': image, 'depth': depth}
        #
        #     if self.transform is not None:
        #         sample_ = self.transform(sample_)  # train val 会通过ToTensor方法将image归一化到[0.0 1.0]之间

        # NOTE: train和val已经离线划分完毕
        # if self.split == 'train' or self.split == 'val' or self.split == 'test':  # or self.split == 'test'
        #     sample = self.data[index]  # 包含image和depth两个路径的列表 并且其路径开头有一个'/'需要去掉
        #     image_path = os.path.join(self.root, sample[0][1:])
        #     depth_path = os.path.join(self.root, sample[1][1:])
        #
        #     image = Image.open(image_path)
        #     image = np.array(image).astype(np.float32)  # RGB图像原本就是[0, 255]之间的，改成浮点型数据
        #     depth = np.load(depth_path, allow_pickle=True)
        #
        #     sample_ = {'image': image, 'depth': depth}
        #     if self.transform is not None:
        #         sample_ = self.transform(sample_)  # train val 会通过ToTensor方法将image归一化到[0.0 1.0]之间
        #
        #     if self.split != 'test':
        #         return sample_
        #     else:
        #         return sample_
        #         # return sample_['image'], sample_['depth']  # 仿照nyu_reduced.py中的做法,测试集直接返回image和depth本身而非数据样本字典

        # NOTE: train和val已经离线划分完毕  在训练集中加入部分测试集
        if self.split == 'train' or self.split == 'val' or self.split == 'test':  # or self.split == 'test'
            sample = self.data[index]  # 包含image和depth两个路径的列表 并且其路径开头有一个'/'需要去掉

            if 'test' in sample[0]:  # 如果是测试集内容
                temp_root = self.root
                temp_root = temp_root.replace('train_dataset', 'test_dataset')
                image_path = os.path.join(temp_root, sample[0][1:])
                depth_path = os.path.join(temp_root, sample[1][1:])
            else:  # 训练集内容
                image_path = os.path.join(self.root, sample[0][1:])
                depth_path = os.path.join(self.root, sample[1][1:])

            image = Image.open(image_path)
            image = np.array(image).astype(np.float32)  # RGB图像原本就是[0, 255]之间的，改成浮点型数据
            depth = np.load(depth_path, allow_pickle=True)

            sample_ = {'image': image, 'depth': depth}
            if self.transform is not None:
                sample_ = self.transform(sample_)  # train val 会通过ToTensor方法将image归一化到[0.0 1.0]之间

            if self.split != 'test':
                return sample_
            else:
                return sample_
                # return sample_['image'], sample_['depth']  # 仿照nyu_reduced.py中的做法,测试集直接返回image和depth本身而非数据样本字典


        # NOTE: 使用bts的稀疏数据集
        # elif self.split == 'test':
        #     data_path = '/home/data/glw/hp/datasets/bts/kitti_dataset/'
        #     gt_path = '/home/data/glw/hp/datasets/bts/kitti_dataset/data_depth_annotated/'
        #
        #     sample_path = self.filenames[index]
        #     image_path = os.path.join(data_path, "./" + sample_path.split()[0])
        #     depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
        #
        #     image = Image.open(image_path)
        #     image = np.array(image).astype(np.float32)  # RGB图像原本就是[0, 255]之间的，改成浮点型数据
        #     depth = Image.open(depth_path)
        #     depth = np.array(depth).astype(np.float32)  # 原本是16位的整型数据
        #     depth = depth / 256.0  # -> meter 可能会超过最大距离80m
        #
        #     sample_ = {'image': image, 'depth': depth}
        #     if self.transform is not None:
        #         sample_ = self.transform(sample_)
        #
        #     return sample_

        # # NOTE: train和val已经离线划分完毕
        # if self.split == 'train' or self.split == 'val' or self.split == 'test':
        #     sample = self.data[index]  # 包含image和depth两个路径的列表 并且其路径开头有一个'/'需要去掉
        #     image_path = os.path.join(self.root, sample[0][1:])
        #     depth_path = os.path.join(self.root, sample[1][1:])
        #
        #     image = Image.open(image_path)
        #     height, width = image.height, image.width
        #
        #     image = np.array(image).astype(np.float32)  # RGB图像原本就是[0, 255]之间的，改成浮点型数据
        #     depth = np.load(depth_path, allow_pickle=True)
        #
        #     # TODO: 在self.transform中取消resize 直接做一个中心裁剪
        #     top_margin = int(height - 352)  # [18, 24]
        #     left_margin = int((width - 1216) / 2)  # [5, 13]
        #     image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
        #     depth = depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]
        #
        #     sample_ = {'image': image, 'depth': depth}
        #     if self.transform is not None:
        #         sample_ = self.transform(sample_)  # train val 会通过ToTensor方法将image归一化到[0.0 1.0]之间
        #
        #     if self.split != 'test':
        #         return sample_
        #     else:
        #         return sample_['image'], sample_['depth']  # 仿照nyu_reduced.py中的做法,测试集直接返回image和depth本身而非数据样本字典

        # NOTE: 从zip文件加载test测试集的情况
        # elif self.split == 'test':
        #     sample = self.data_indices[index]
        #     sample[0] = 'test_dataset' + sample[0]
        #     sample[1] = 'test_dataset' + sample[1]
        #     image = Image.open(BytesIO(self.data[sample[0]]))
        #     depth = np.load(BytesIO(self.data[sample[1]]), allow_pickle=True)  # 已经是meter单位的深度数据
        #     image = np.array(image).astype(np.float32)  # RGB图像原本就是[0, 255]之间的，改成浮点型数据
        #     image = torch.from_numpy(image).type(torch.float32)  # [0.0, 255.0]
        #     depth = torch.from_numpy(depth).type(torch.float32)  # [0.0, 80.0]
        #     # print('image.shape:{} depth.shape:{}'.format(image.shape, depth.shape))
        #
        #     sample_ = {'image': image, 'depth': depth}  # train val
        #
        #     if self.transform is not None:
        #         sample_ = self.transform(sample_)  # test并没有做任何处理
        #     # 仿照nyu_reduced.py中的做法,测试集直接返回image和depth本身而非数据样本字典
        #     return sample_['image'], sample_['depth']

    def __len__(self):
        return len(self.data)

    def train_transform(self, data):  # train训练集
        if self.augmentation == 'alhashim':
            transform = Compose([
                Resize(self.resolution),
                RandomHorizontalFlip(),
                RandomChannelSwap(0.25),
                ToTensor(test=False, maxDepth=80.0)
            ])
        else:
            transform = Compose([
                RandomRotation(4.5),
                CenterCrop(self.resolution),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomChannelSwap(0.25),
                ToTensor(test=False, maxDepth=80.0)
            ])

        data = transform(data)
        return data

    def val_transform(self, data):  # validation验证集
        if self.augmentation == 'alhashim':
            transform = Compose([
                Resize(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])
        else:
            transform = Compose([
                CenterCrop(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])

        data = transform(data)
        return data

    def test_transform(self, data):  # test测试集
        if self.augmentation == 'alhashim':
            transform = Compose([
                Resize(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])
        else:
            transform = Compose([
                Resize(self.resolution),
                ToTensor(test=True, maxDepth=80.0)
            ])

        data = transform(data)
        return data

    def normalize2img(self, x):
        vmax = np.max(x)
        vmin = np.min(x)
        return ((x - vmin) * 255.0 / vmax).astype(np.float32)


def LoadZipToMemKitti(zip_file, mode='train'):
    """
    NOTE: 在初始化Dataloader之前调用
    Args:
        zip_file:
        mode:
    Returns:
    """
    # Load zip file into memory
    print('Loading {} dataset zip file...'.format(mode))
    t0 = time.time()
    input_zip = ZipFile(zip_file)
    # data是一个字典 它的key是zip中每一级路径名字 value就是每个路径下对应的文件
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    print('It took {:.1f}min to load and read the zip file.'.format((time.time() - t0) / 60))

    # data_indices是一个list，其中包含了n个list，这每个list都分别包含image和对应的depth文件路径
    if mode == 'train':
        data_indices = list(
            (row.split(',') for row in (data['train_dataset/train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    elif mode == 'test':
        data_indices = list(
            (row.split(',') for row in (data['test_dataset/test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    for i in range(len(data_indices)):
        if '\r' in data_indices[i][1]:
            data_indices[i][1] = data_indices[i][1].replace('\r', '')

    # for key in data.keys():
    #     print('data.key:{}'.format(key))
    # for i in range(len(data_indices)):
    #     print('data_indices[{}]: {}'.format(i, data_indices[i]))
    # exit()

    print('Loaded {} images for {}.'.format(len(data_indices), mode))
    return data, data_indices


def get_kitti_dataset(path, mode='train', resolution='full', augmentation='alhashim'):
    """
    Args:
        zip_file:
        mode: 'train'——在线分割训练集和验证集 'test'——测试集用于evaluation
    Returns: torch.utils.data.DataLoader
    """
    data, data_indices = LoadZipToMemKitti(zip_file=path, mode=mode)

    if mode == 'train':
        # TODO
        sample_nums = len(data_indices)
        # val_nums = int(sample_nums * 0.2)  # 百分比
        val_nums = sample_nums - 20000  # Eigen论文中指定训练集样本数为20K
        print('{} samples will be used for validation.'.format(val_nums))
        val_indices = random.sample(range(sample_nums), val_nums)
        train_indices = [i for i in range(sample_nums) if i not in val_indices]
        train_dataset = KITTIDataset(path, data, train_indices, 'train', resolution, augmentation)
        test_dataset = KITTIDataset(path, data, val_indices, 'val', resolution, augmentation)
        return [train_dataset, test_dataset]  # 返回一个list

    elif mode == 'test':
        test_dataset = KITTIDataset(path, data, data_indices, mode, resolution, augmentation)
        return test_dataset


def get_kitti_dataset_(path, mode='train', resolution='full', augmentation='alhashim'):
    """
    Args:
        path: 如果是训练集 则路径需要到/train_dataset 如果是测试集 则路径需要到/test_dataset.zip
        mode: 'train'——在线分割训练集和验证集 'test'——测试集用于evaluation
        resolution:
        augmentation:

    Returns: KITTIDataset
    """
    # NOTE: 在线划分train和val
    # if mode == 'train':
    #     data_path_list = []
    #     with open(os.path.join(path, 'train.csv'), 'r') as file:
    #         reader = csv.reader(file)
    #         # 逐行读取CSV文件内容
    #         for row in reader:
    #             # 将每一行数据添加到列表中
    #             data_path_list.append(row)
    #
    #     sample_nums = len(data_path_list)
    #     # val_nums = int(sample_nums * 0.2)  # 百分比
    #     train_nums = 20000  # Eigen论文中指定训练集样本数为20K
    #     val_nums = sample_nums - train_nums
    #     print('{}, {} samples will be used for training and validation respectively.'.format(train_nums, val_nums))
    #
    #     val_indices = random.sample(range(sample_nums), val_nums)
    #     train_indices = [i for i in range(sample_nums) if i not in val_indices]
    #
    #     # temp1 = data_path_list[train_indices[0]][0][1:]
    #     # temp2 = data_path_list[val_indices[0]][0][1:]
    #     # res1 = os.path.join(path, temp1)
    #     # res2 = os.path.join(path, temp2)
    #     # print('path:{}'.format(path))
    #     # print('temp1:{}'.format(temp1))
    #     # print('temp2:{}'.format(temp2))
    #     # print('res1:{}'.format(res1))
    #     # print('res2:{}'.format(res2))
    #     # exit()
    #
    #     train_dataset = KITTIDataset_(path, data_path_list, train_indices, 'train', resolution, augmentation)
    #     test_dataset = KITTIDataset_(path, data_path_list, val_indices, 'val', resolution, augmentation)
    #     return [train_dataset, test_dataset]  # 返回一个list

    # NOTE: train和val已经离线划分完毕
    if mode == 'train':
        train_path_list = []
        with open(os.path.join(path, 'train_.csv'), 'r') as file:
            reader = csv.reader(file)
            # 逐行读取CSV文件内容
            for row in reader:
                # 将每一行数据添加到列表中
                train_path_list.append(row)
        print('there are {} samples in train dataset.'.format(len(train_path_list)))
        train_dataset = KITTIDataset_(path, train_path_list, None, 'train', resolution, augmentation)
        return train_dataset

    elif mode == 'val':
        val_path_list = []
        with open(os.path.join(path, 'val_.csv'), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                val_path_list.append(row)
        print('there are {} samples in val dataset.'.format(len(val_path_list)))
        val_dataset = KITTIDataset_(path, val_path_list, None, 'val', resolution, augmentation)
        return val_dataset

    # elif mode == 'test':
    #     data, data_indices = LoadZipToMemKitti(zip_file=path, mode='test')
    #     test_dataset = KITTIDataset_(path, data, data_indices, 'test', resolution, augmentation)
    #     return test_dataset

    elif mode == 'test':
        test_path_list = []
        with open(os.path.join(path, 'test.csv'), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                test_path_list.append(row)
        print('there are {} samples in test dataset.'.format(len(test_path_list)))
        test_dataset = KITTIDataset_(path, test_path_list, None, 'test', resolution, augmentation)
        return test_dataset


    # NOTE: 重构逻辑
    # if mode == 'train':
    #     file_path = os.path.join(path, 'train_.csv')
    # elif mode == 'val':
    #     file_path = os.path.join(path, 'val_.csv')
    # elif mode == 'test':
    #     file_path = os.path.join(path, 'test.csv')
    #
    # sample_path_list = []
    # with open(file_path, 'r') as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         sample_path_list.append(row)
    #
    # dataset = KITTIDataset_(path, sample_path_list, None, mode, resolution, augmentation)
    # return dataset


if __name__ == '__main__':
    # root = '/home/data/glw/hp/datasets/kitti_eigen/test_dataset.zip'
    # LoadZipToMemKitti(root, mode='test')
    # exit()

    path = '/home/data/glw/hp/datasets/kitti_eigen/train_dataset'
    get_kitti_dataset_(path, mode='train')









