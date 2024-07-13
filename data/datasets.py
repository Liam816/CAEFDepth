import os

from torch.utils.data import DataLoader

from data.kitti import KITTIDataset, get_kitti_dataset, get_kitti_dataset_
from data.nyu_reduced import get_NYU_dataset

"""
Preparation of dataloaders for Datasets
"""


def get_dataloader(dataset_name,
                   model_name,
                   path,
                   split='train',
                   resolution='full',
                   augmentation='alhashim',
                   interpolation='linear',
                   batch_size=8,
                   workers=4,
                   uncompressed=True,
                   train_edge=False,
                   use_depthnorm=True):
    if dataset_name == 'kitti':
        # dataset = get_kitti_dataset(path,
        #                             split,
        #                             resolution,
        #                             augmentation)
        dataset = get_kitti_dataset_(path,
                                     split,
                                     resolution,
                                     augmentation)

    elif dataset_name == 'nyu_reduced':
        dataset = get_NYU_dataset(path,
                                  split,
                                  resolution=resolution,
                                  uncompressed=uncompressed,
                                  train_edge=train_edge,
                                  use_depthnorm=use_depthnorm)
    else:
        print('Dataset does not exist.')
        exit(0)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=(split == 'train'),
                            num_workers=workers,
                            pin_memory=True)
    return dataloader
