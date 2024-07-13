import numpy as np
from PIL import Image
import random
import os
import pandas as pd


img_root = '/home/ping.he/projects/MDE/dataset/nyu/nyu_test/rgb_img/'
depth_root = '/home/ping.he/projects/MDE/dataset/nyu/nyu_test/depth/'


def parse_testset_and_generate_imgs():
    rgb_np = np.load('/home/ping.he/projects/MDE/dataset/nyu/nyu_test/eigen_test_rgb.npy')  # [0.0, 255.0]
    depth_np = np.load('/home/ping.he/projects/MDE/dataset/nyu/nyu_test/eigen_test_depth.npy')  # [0.0, 10.0]
    rgb_np = rgb_np.astype(np.uint8)
    print('rgb_np shape:', rgb_np.shape)
    print('rgb_np max:{} min:{}'.format(np.max(rgb_np), np.min(rgb_np)))
    depth_np = (depth_np / 10.0 * 255.0).astype(np.uint8)
    print('depth_np shape:', depth_np.shape)
    print('depth_np max:{} min:{}'.format(np.max(depth_np), np.min(depth_np)))

    for i in range(rgb_np.shape[0]):
        image = Image.fromarray(rgb_np[i])
        image.save(img_root + 'test_{}.jpg'.format(i), 'JPEG')
        # exit()

    for i in range(depth_np.shape[0]):
        image = Image.fromarray(depth_np[i], 'L')
        image.save(depth_root + 'test_{}.png'.format(i))
        # exit()   


def generate_new_train_csv(ratio=1.0):
    img_files = []
    dst_root = 'data/testset_for_train/'
    for dirpath, dirnames, files in os.walk(img_root):
        for file in files:
            img_files.append(dst_root + file)
    depth_files = []
    for _, _, files in os.walk(img_root):
        for file in files:
            depth_files.append(dst_root + file)
    # print('img_files:', img_files[:10])
    # print('depth_files:', depth_files[:10])
    data = []
    for img, depth in zip(img_files, depth_files):
        data.append([img, depth])
    random.seed(816)
    random.shuffle(data)
    sample_num = int(len(data) * ratio)
    data = data[:sample_num]

    raw_csv = '/home/ping.he/projects/MDE/dataset/nyu/nyu_data/data/nyu2_train_raw.csv'
    df = pd.read_csv(raw_csv, nrows=50688, header=None)
    df_to_append = pd.DataFrame(data)
    df = pd.concat([df, df_to_append], ignore_index=True)
    dst_csv = '/home/ping.he/projects/MDE/dataset/nyu/nyu_data/data/nyu2_train.csv'
    df.to_csv(dst_csv, index=False, header=False)


if __name__ == '__main__':
    # parse_testset_and_generate_imgs()
    # exit()

    generate_new_train_csv()

