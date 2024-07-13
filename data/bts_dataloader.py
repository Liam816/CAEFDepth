import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
import torchvision
from torchvision import transforms
from PIL import Image
import os
import random

# from distributed_sampler_no_evenly_divisible import *

import multiprocessing
import matplotlib.pyplot as plt


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class BtsDataLoader(object):
    def __init__(self, opts, mode):
        train_edge = getattr(opts, "common.train_edge", False)
        use_liam_dataset = getattr(opts, "bts.use_liam_dataset", False)
        print('use_liam_dataset:{}'.format(use_liam_dataset))

        if mode == 'train':
            if use_liam_dataset:
                # print('using DataLoadPreprocessLIAM')
                # exit()
                self.training_samples = DataLoadPreprocessLIAM(opts, mode, transform=preprocessing_transforms(mode))
            else:
                # print('using DataLoadPreprocess')
                # exit()
                self.training_samples = DataLoadPreprocess(opts, mode, transform=preprocessing_transforms(mode),
                                                           train_edge=train_edge)

            self.train_sampler = None

            batch_size = getattr(opts, "common.bs", 8)
            n_cpus = multiprocessing.cpu_count()
            num_workers = n_cpus // 2
            # print('batch_size: {}'.format(batch_size))
            self.data = DataLoader(self.training_samples, batch_size=batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            if use_liam_dataset:
                self.testing_samples = DataLoadPreprocessLIAM(opts, mode, transform=preprocessing_transforms(mode))
            else:
                self.testing_samples = DataLoadPreprocess(opts, mode, transform=preprocessing_transforms(mode),
                                                          train_edge=train_edge)

            self.eval_sampler = None

            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            if use_liam_dataset:
                self.testing_samples = DataLoadPreprocessLIAM(opts, mode, transform=preprocessing_transforms(mode))
            else:
                self.testing_samples = DataLoadPreprocess(opts, mode, transform=preprocessing_transforms(mode),
                                                          train_edge=train_edge)

            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            

# NOTE: 原版
# class DataLoadPreprocess(Dataset):
#     def __init__(self, opts, mode, transform=None, is_for_online_eval=False, train_edge=False):
#         self.opts = opts
#         self.dataset = getattr(opts, 'bts.dataset')
#         self.use_right = getattr(opts, 'bts.use_right')
#         self.data_path = getattr(opts, 'bts.data_path')
#         self.gt_path = getattr(opts, 'bts.gt_path')
#         self.do_kb_crop = getattr(opts, 'bts.do_kb_crop')
#         self.do_random_rotate = getattr(opts, 'bts.do_random_rotate')
#         self.degree = getattr(opts, 'bts.degree', 1.0)
#         self.input_height = getattr(opts, 'bts.input_height')
#         self.input_width = getattr(opts, 'bts.input_width')
#         self.data_path_eval = getattr(opts, 'bts.data_path_eval')
#         self.gt_path_eval = getattr(opts, 'bts.gt_path_eval')
#         self.train_edge = train_edge
#
#         if mode == 'online_eval':  # validation (evaluation)
#             filenames_file_eval = getattr(opts, 'bts.filenames_file_eval')
#             with open(filenames_file_eval, 'r') as f:  # 697个样本 但其中有部分gt深度图为None的情况 将这部分数据已经去除
#                 self.filenames = f.readlines()
#         else:  # train
#             filenames_file = getattr(opts, 'bts.filenames_file')
#             with open(filenames_file, 'r') as f:  # 23158个样本
#                 self.filenames = f.readlines()  # 一行数据包含了image、depth（格式都是.png）和焦距
#
#         self.mode = mode
#         self.transform = transform
#         self.to_tensor = ToTensor
#         self.is_for_online_eval = is_for_online_eval
#
#     def __getitem__(self, idx):
#         sample_path = self.filenames[idx]  # 包含了image、depth（格式都是.png）和焦距
#         focal = float(sample_path.split()[2])  # 按照空格分割
#
#         if self.mode == 'train':
#             if self.dataset == 'kitti' and self.use_right is True and random.random() > 0.5:
#                 image_path = os.path.join(self.data_path, "./" + sample_path.split()[3])
#                 depth_path = os.path.join(self.gt_path, "./" + sample_path.split()[4])
#             else:  # 默认为该种情况
#                 image_path = os.path.join(self.data_path, sample_path.split()[0])
#                 depth_path = os.path.join(self.gt_path, sample_path.split()[1])
#
#             image = Image.open(image_path)  # 每张图片尺寸不一定一致 会相差几个像素
#             depth_gt = Image.open(depth_path)  # depth_gt: PIL.image [0, 22031+](单位 meter*256)
#
#             # print('1 type(depth_gt):{}'.format(type(depth_gt)))
#             # print('1 max:{} min:{}'.format(np.max(depth_gt), np.min(depth_gt)))
#             # exit()
#
#             if self.do_kb_crop is True:  # 默认为True
#                 height = image.height
#                 width = image.width
#                 top_margin = int(height - 352)  # [18, 24]
#                 left_margin = int((width - 1216) / 2)  # [5, 13]
#                 depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))  # [352, 1216]
#                 image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))  # [352, 1216]
#
#             # To avoid blank boundaries due to pixel registration
#             if self.dataset == 'nyu':
#                 depth_gt = depth_gt.crop((43, 45, 608, 472))
#                 image = image.crop((43, 45, 608, 472))
#
#             if self.do_random_rotate is True:  # 默认为True
#                 random_angle = (random.random() - 0.5) * 2 * self.degree
#                 image = self.rotate_image(image, random_angle)
#                 depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
#
#             image = np.asarray(image, dtype=np.float32) / 255.0  # [0, 255] -> [0, 1]
#             depth_gt = np.asarray(depth_gt, dtype=np.float32)
#             depth_gt = np.expand_dims(depth_gt, axis=2)
#
#             if self.dataset == 'nyu':
#                 depth_gt = depth_gt / 1000.0
#             else:  # kitti
#                 depth_gt = depth_gt / 256.0  # ->meter 可能会超过最大距离80m  depth_gt: numpy.ndarray [0, 80.0+](单位 meters)
#
#             # print('2 type(depth_gt):{}'.format(type(depth_gt)))
#             # print('2 max:{} min:{}'.format(np.max(depth_gt), np.min(depth_gt)))
#             # exit()
#
#             # NOTE: 参照GuideDepth训练时需要将gt深度值转为DepthNorm
#             # zero_mask = depth_gt == 0.0
#             # depth_gt = np.clip(depth_gt, 0.8, 80)
#             # depth_gt = 80 / depth_gt
#             # depth_gt[zero_mask] = 0.0
#             # depth_gt = np.expand_dims(depth_gt, axis=2)
#
#             # NOTE: 与BTS论文中不同 直接将352×1216分辨率的图片送入网络训练
#             # image, depth_gt = self.random_crop(image, depth_gt, self.input_height, self.input_width)
#             image, depth_gt = self.train_preprocess(image, depth_gt)  # depth_gt: numpy.ndarray [0, 80.0+](单位 meters)
#
#             # print('3 type(depth_gt):{}'.format(type(depth_gt)))
#             # print('3 max:{} min:{}'.format(np.max(depth_gt), np.min(depth_gt)))
#             # exit()
#
#             sample = {'image': image, 'depth': depth_gt, 'focal': focal}
#
#         else:  # test 测试集
#             if self.mode == 'online_eval':
#                 data_path = self.data_path_eval
#             else:
#                 data_path = self.data_path
#
#             image_path = os.path.join(data_path, "./" + sample_path.split()[0])
#             image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
#
#             if self.mode == 'online_eval':
#                 gt_path = self.gt_path_eval
#                 depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
#                 has_valid_depth = False
#                 try:
#                     depth_gt = Image.open(depth_path)
#                     has_valid_depth = True
#                 except IOError:
#                     depth_gt = False
#                     # print('Missing gt for {}'.format(image_path))
#
#                 if has_valid_depth:
#                     depth_gt = np.asarray(depth_gt, dtype=np.float32)
#                     depth_gt = np.expand_dims(depth_gt, axis=2)
#                     if self.dataset == 'nyu':
#                         depth_gt = depth_gt / 1000.0
#                     else:
#                         depth_gt = depth_gt / 256.0  # ->meter 可能会超过最大距离80m
#
#             if self.do_kb_crop is True:  # 默认为True
#                 height = image.shape[0]
#                 width = image.shape[1]
#                 top_margin = int(height - 352)
#                 left_margin = int((width - 1216) / 2)
#                 image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]  # [352, 1216]
#                 if self.mode == 'online_eval' and has_valid_depth:
#                     depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]  # [352, 1216]
#
#             if self.mode == 'online_eval':
#                 sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
#             else:
#                 sample = {'image': image, 'focal': focal}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
#
#     def rotate_image(self, image, angle, flag=Image.BILINEAR):
#         result = image.rotate(angle, resample=flag)
#         return result
#
#     def random_crop(self, img, depth, height, width):
#         assert img.shape[0] >= height
#         assert img.shape[1] >= width
#         assert img.shape[0] == depth.shape[0]
#         assert img.shape[1] == depth.shape[1]
#         x = random.randint(0, img.shape[1] - width)
#         y = random.randint(0, img.shape[0] - height)
#         img = img[y:y + height, x:x + width, :]
#         depth = depth[y:y + height, x:x + width, :]
#         return img, depth
#
#     def train_preprocess(self, image, depth_gt):
#         # Random flipping
#         do_flip = random.random()
#         if do_flip > 0.5:
#             image = (image[:, ::-1, :]).copy()
#             depth_gt = (depth_gt[:, ::-1, :]).copy()
#
#         # Random gamma, brightness, color augmentation
#         do_augment = random.random()
#         if do_augment > 0.5:
#             image = self.augment_image(image)
#
#         return image, depth_gt
#
#     def augment_image(self, image):
#         # gamma augmentation
#         gamma = random.uniform(0.9, 1.1)
#         image_aug = image ** gamma
#
#         # brightness augmentation
#         if self.dataset == 'nyu':
#             brightness = random.uniform(0.75, 1.25)
#         else:
#             brightness = random.uniform(0.9, 1.1)
#         image_aug = image_aug * brightness
#
#         # color augmentation
#         colors = np.random.uniform(0.9, 1.1, size=3)
#         white = np.ones((image.shape[0], image.shape[1]))
#         color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
#         image_aug *= color_image
#         image_aug = np.clip(image_aug, 0, 1)
#
#         return image_aug
#
#     def __len__(self):
#         return len(self.filenames)


class DataLoadPreprocess(Dataset):
    def __init__(self, opts, mode, transform=None, is_for_online_eval=False, train_edge=False):
        self.opts = opts
        self.dataset = getattr(opts, 'bts.dataset')
        self.use_right = getattr(opts, 'bts.use_right')
        self.data_path = getattr(opts, 'bts.data_path')
        self.gt_path = getattr(opts, 'bts.gt_path')
        self.do_kb_crop = getattr(opts, 'bts.do_kb_crop')
        self.do_random_rotate = getattr(opts, 'bts.do_random_rotate')
        self.degree = getattr(opts, 'bts.degree', 1.0)
        self.input_height = getattr(opts, 'bts.input_height')
        self.input_width = getattr(opts, 'bts.input_width')
        self.data_path_eval = getattr(opts, 'bts.data_path_eval')
        self.gt_path_eval = getattr(opts, 'bts.gt_path_eval')
        self.train_edge = train_edge

        if mode == 'online_eval':  # validation (evaluation)
            filenames_file_eval = getattr(opts, 'bts.filenames_file_eval')
            with open(filenames_file_eval, 'r') as f:  # 697个样本 但其中有部分gt深度图为None的情况 将这部分数据已经去除
                self.filenames = f.readlines()
        else:  # train
            filenames_file = getattr(opts, 'bts.filenames_file')
            with open(filenames_file, 'r') as f:  # 23158个样本
                self.filenames = f.readlines()  # 一行数据包含了image、depth（格式都是.png）和焦距

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        if train_edge:
            self.edge_transformation = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]  # 包含了image、depth（格式都是.png）和焦距
        focal = float(sample_path.split()[2])  # 按照空格分割

        if self.mode == 'train':
            if self.dataset == 'kitti' and self.use_right is True and random.random() > 0.5:
                image_path = os.path.join(self.data_path, "./" + sample_path.split()[3])
                depth_path = os.path.join(self.gt_path, "./" + sample_path.split()[4])
            else:  # 默认为该种情况
                image_path = os.path.join(self.data_path, sample_path.split()[0])
                depth_path = os.path.join(self.gt_path, sample_path.split()[1])

            image = Image.open(image_path)  # 每张图片尺寸不一定一致 会相差几个像素
            depth_gt = Image.open(depth_path)  # depth_gt: PIL.image [0, 22031+](单位 meter*256)

            if self.train_edge:
                root = os.path.join(self.data_path, 'data_depth_edge_gt', 'train_edge_gt')
                scene = sample_path.split()[0].split('/')[1]
                sn = sample_path.split()[0].split('/')[-1].split('.')[0]
                depth_edge = Image.open(os.path.join(root, scene, sn + '_edge.png'))
                # print('depth_edge h:{} w:{}'.format(depth_edge.height, depth_edge.width))
                # depth_edge = np.array(depth_edge).astype(np.uint8)  # [h, w]
                # depth_edge = self.edge_transformation(depth_edge)  # numpy [h, w] -> tensor [1, h, w]

            if self.do_kb_crop is True:  # 默认为True
                height = image.height
                width = image.width
                top_margin = int(height - 352)  # [18, 24]
                left_margin = int((width - 1216) / 2)  # [5, 13]
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))  # [352, 1216]
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))  # [352, 1216]
                if self.train_edge:
                    depth_edge = depth_edge.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))  # [352, 1216]
                    depth_edge = np.array(depth_edge).astype(np.uint8)  # [h, w]
                    depth_edge = self.edge_transformation(depth_edge)  # numpy [h, w] -> tensor [1, h, w]
                    # print('depth_edge.shape:{}'.format(depth_edge.shape))

            # To avoid blank boundaries due to pixel registration
            if self.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))

            if self.do_random_rotate is True:  # 默认为True
                random_angle = (random.random() - 0.5) * 2 * self.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0  # [0, 255] -> [0, 1]
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:  # kitti
                depth_gt = depth_gt / 256.0  # ->meter 可能会超过最大距离80m  depth_gt: numpy.ndarray [0, 80.0+](单位 meters)

            # NOTE: 与BTS论文中不同 直接将352×1216分辨率的图片送入网络训练
            # image, depth_gt = self.random_crop(image, depth_gt, self.input_height, self.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)  # depth_gt: numpy.ndarray [0, 80.0+](单位 meters)

            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        else:  # test 测试集
            if self.mode == 'online_eval':
                data_path = self.data_path_eval
            else:
                data_path = self.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0  # ->meter 可能会超过最大距离80m

            if self.do_kb_crop is True:  # 默认为True
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]  # [352, 1216]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]  # [352, 1216]
                if self.train_edge:
                    root = os.path.join(self.data_path, 'data_depth_edge_gt', 'test_edge_gt')
                    scene = sample_path.split()[0].split('/')[1]
                    sn = sample_path.split()[0].split('/')[-1].split('.')[0]
                    depth_edge = Image.open(os.path.join(root, scene, sn + '_edge.png'))
                    depth_edge = np.array(depth_edge).astype(np.uint8)  # [h, w]
                    depth_edge = depth_edge[top_margin:top_margin + 352, left_margin:left_margin + 1216]  # [352, 1216]
                    depth_edge = self.edge_transformation(depth_edge)  # numpy [h, w] -> tensor [1, h, w]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        if self.train_edge:
            sample['depth_edge'] = depth_edge

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


# NOTE: 加载colorize方法填充后的稠密深度图
class DataLoadPreprocessLIAM(Dataset):
    def __init__(self, opts, mode, transform=None, is_for_online_eval=False):
        self.opts = opts
        self.dataset = getattr(opts, 'bts.dataset')
        self.use_right = getattr(opts, 'bts.use_right')
        self.data_path = getattr(opts, 'bts.data_path')
        self.gt_path = getattr(opts, 'bts.gt_path')
        self.do_kb_crop = getattr(opts, 'bts.do_kb_crop')
        self.do_random_rotate = getattr(opts, 'bts.do_random_rotate')
        self.degree = getattr(opts, 'bts.degree', 1.0)
        self.input_height = getattr(opts, 'bts.input_height')
        self.input_width = getattr(opts, 'bts.input_width')
        self.data_path_eval = getattr(opts, 'bts.data_path_eval')
        self.gt_path_eval = getattr(opts, 'bts.gt_path_eval')

        if mode == 'online_eval':  # validation (evaluation)
            filenames_file_eval = getattr(opts, 'bts.filenames_file_eval')
            with open(filenames_file_eval, 'r') as f:  # 697个样本 但其中有部分gt深度图为None的情况 将这部分数据已经去除
                self.filenames = f.readlines()
        else:  # train
            filenames_file = getattr(opts, 'bts.filenames_file')
            with open(filenames_file, 'r') as f:  # 23158个样本
                self.filenames = f.readlines()  # 一行数据包含了image、depth（格式都是.png）和焦距

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]  # 包含了image、depth（格式都是.png）和焦距
        focal = float(sample_path.split()[2])  # 按照空格分割

        if self.mode == 'train':
            image_path = os.path.join(self.data_path, sample_path.split()[0])
            scene, img_name = sample_path.split()[0].split('/')[1], sample_path.split()[0].split('/')[-1].split('.')[0]
            self.gt_path = '/home/data/glw/hp/datasets/bts/data_depth_dense_filled/'
            depth_path = os.path.join(self.gt_path, 'train', scene, img_name + '_dense_filled_1.npy')
            if not os.path.exists(depth_path):
                print('{} dose not exist.'.format(depth_path))

            image = Image.open(image_path)  # 每张图片尺寸不一定一致 会相差几个像素
            depth_gt = np.load(depth_path, allow_pickle=True)  # [h, w]
            a, b = np.max(depth_gt), np.min(depth_gt)
            # print('0 type(depth_gt):{}'.format(type(depth_gt)))
            # print('0 max:{} min:{}'.format(np.max(depth_gt), np.min(depth_gt)))
            depth_gt = Image.fromarray(np.uint8(depth_gt))

            if self.do_kb_crop is True:  # 默认为True
                height = image.height
                width = image.width
                top_margin = int(height - 352)  # [18, 24]
                left_margin = int((width - 1216) / 2)  # [5, 13]
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))  # [352, 1216]
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))  # [352, 1216]

            # To avoid blank boundaries due to pixel registration
            if self.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))

            if self.do_random_rotate is True:  # 默认为True
                random_angle = (random.random() - 0.5) * 2 * self.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0  # [0, 255] -> [0, 1]
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            # NOTE: 从numpy数组转到PIL图片以后值域自动变成了[0, 255]
            c, d = np.max(depth_gt), np.min(depth_gt)
            depth_gt = (depth_gt - d) * (a - b) / (c - d) + b
            # print('1 type(depth_gt):{}'.format(type(depth_gt)))
            # print('1 max:{} min:{}'.format(np.max(depth_gt), np.min(depth_gt)))

            if self.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:  # kitti
                depth_gt = depth_gt / 256.0  # ->meter 可能会超过最大距离80m

            # print('2 type(depth_gt):{}'.format(type(depth_gt)))
            # print('2 max:{} min:{}'.format(np.max(depth_gt), np.min(depth_gt)))

            # NOTE: 参照GuideDepth训练时需要将gt深度值转为DepthNorm
            # zero_mask = depth_gt == 0.0
            # depth_gt = np.clip(depth_gt, 0.8, 80)
            # depth_gt = 80 / depth_gt
            # depth_gt[zero_mask] = 0.0
            # depth_gt = np.expand_dims(depth_gt, axis=2)

            # NOTE: 与BTS论文中不同 直接将352×1216分辨率的图片送入网络训练
            # image, depth_gt = self.random_crop(image, depth_gt, self.input_height, self.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)

            # print('3 type(depth_gt):{}'.format(type(depth_gt)))
            # print('3 max:{} min:{}'.format(np.max(depth_gt), np.min(depth_gt)))
            # exit()

            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        else:  # test 测试集
            if self.mode == 'online_eval':
                data_path = self.data_path_eval
            else:
                data_path = self.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.gt_path_eval
                # depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                scene, img_name = sample_path.split()[0].split('/')[1], \
                                  sample_path.split()[0].split('/')[-1].split('.')[0]
                gt_path = '/home/data/glw/hp/datasets/bts/data_depth_dense_filled/'
                depth_path = os.path.join(gt_path, 'test', scene, img_name + '_dense_filled_1.npy')
                if not os.path.exists(depth_path):
                    print('{} dose not exist.'.format(depth_path))

                has_valid_depth = False
                try:
                    # depth_gt = Image.open(depth_path)
                    # depth_gt = Image.fromarray(np.uint8(depth_gt))
                    depth_gt = np.load(depth_path, allow_pickle=True)  # [h, w]
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    # depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.float32(depth_gt)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0  # ->meter 可能会超过最大距离80m

            if self.do_kb_crop is True:  # 默认为True
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]  # [352, 1216]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]  # [352, 1216]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':  # or self.mode == 'online_eval'
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
