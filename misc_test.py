import numpy as np
from zipfile import ZipFile
from io import BytesIO
import torch
from model.DDRNet_23_slim import DualResNet_Backbone
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize
from torchvision import transforms
from nets.models.MobileViT_GUB import MobileViTGUB
import shutil
from torch import nn, Tensor
import os
import glob
import copy

import torch.cuda

from data import datasets
import matplotlib.pyplot as plt
from data.bts_dataloader import BtsDataLoader
from utils.main_utils import parse_arguments, load_config_file

import random
from timm.models import create_model
from torchvision import models

# from pynvml import *
#
#
# def show_gpu(simlpe=True):
#     # 初始化
#     nvmlInit()
#     # 获取GPU个数
#     deviceCount = nvmlDeviceGetCount()
#     total_memory = 0
#     total_free = 0
#     total_used = 0
#     gpu_name = ""
#     gpu_num = deviceCount
#
#     for i in range(deviceCount):
#         handle = nvmlDeviceGetHandleByIndex(i)
#         info = nvmlDeviceGetMemoryInfo(handle)
#         gpu_name = nvmlDeviceGetName(handle).decode('utf-8')
#         # 查看型号、显存、温度、电源
#         if not simlpe:
#             print("[ GPU{}: {}".format(i, gpu_name), end="    ")
#             print("总共显存: {}G".format((info.total // 1048576) / 1024), end="    ")
#             print("空余显存: {}G".format((info.free // 1048576) / 1024), end="    ")
#             print("已用显存: {}G".format((info.used // 1048576) / 1024), end="    ")
#             print("显存占用率: {}%".format(info.used / info.total), end="    ")
#             print("运行温度: {}摄氏度 ]".format(nvmlDeviceGetTemperature(handle, 0)))
#
#         total_memory += (info.total // 1048576) / 1024
#         total_free += (info.free // 1048576) / 1024
#         total_used += (info.used // 1048576) / 1024
#
#     print("显卡名称：[{}]，显卡数量：[{}]，总共显存；[{}G]，空余显存：[{}G]，已用显存：[{}G]，显存占用率：[{}%]。".format(gpu_name, gpu_num, total_memory,
#                                                                                      total_free, total_used,
#                                                                                      (total_used / total_memory)))
#
#     # 关闭管理工具
#     nvmlShutdown()

# gpu_nums = torch.cuda.device_count()
# print('gpu_nums:{}'.format(gpu_nums))

# # 原始二维数组
# arr = np.array([[1, 2], [3, 4]])
#
# # 在前面插入两个新的维度
# new_arr = np.expand_dims(arr, axis=0)
# new_arr = np.expand_dims(new_arr, axis=0)
# print(new_arr.shape)  # 输出: (1, 2, 2)
#
# exit()
#
# x = torch.arange(1, 13)
# y = x.reshape(1, 2, 3, 2)
# z = y.data
# print('type(z):{}'.format(type(z)))
# print('z:', z)
# exit()

net = models.resnet18(pretrained=False)
max_epoch = 50  # 一共50 epoch
iters = 200  # 每个epoch 有 200 个 bach

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=max_epoch)  # * iters

lr = []
for epoch in range(max_epoch):
    for batch in range(iters):
        optimizer.step()

        lr.append(scheduler.get_lr()[0])
    scheduler.step()  # 注意 每个epoch 结束， 更新learning rate

plt.plot(np.arange(len(lr)), lr)
plt.show()
print('hello world')
exit()


# To Train from scratch/fine-tuning
model = create_model("fastvit_t8")
print('model created.')
exit()


# tensor = torch.tensor([1, 2, float('nan'), 4, float('nan')])
tensor = torch.randn(8, 3, 20, 20)
# 随机指定四个元素为NaN值
for _ in range(4):
    i = random.randint(0, 7)
    j = random.randint(0, 2)
    x = random.randint(0, 19)
    y = random.randint(0, 19)
    print('i:{} j:{} x:{} y:{}'.format(i, j, x, y))
    tensor[i, j, x, y] = float('nan')

has_nan = torch.isnan(tensor)
nan_indices = torch.nonzero(has_nan)

if has_nan.any():
    print("张量中存在NaN值")
    print("NaN值的位置：", nan_indices)
else:
    print("张量中不存在NaN值")

exit()


# 创建一个二维卷积层
conv_op = nn.Conv2d(3, 3, 3, padding=(1, 1), bias=False)

# 获取卷积层的参数张量
params = list(conv_op.parameters())

# 打印参数形状和是否需要梯度
for param in params:
    print(param.shape)
    print(param.requires_grad)

exit()

x = torch.arange(1, 25)
x = torch.reshape(x, (2, 3, 2, 2))
print('x:\n', x)
b, c, h, w = x.shape
x = x.reshape(b, c, h * w)
vmax = torch.max(x, dim=-1, keepdim=True)
print('vmax.values.shape:{}'.format(vmax.values.shape))
vmax = vmax.values
print('vmax.shape:{}'.format(vmax.shape))
vmax = vmax.repeat(1, 1, h * w)
vmax = vmax.reshape(b, c, h, w)
print('vmax.shape:{}'.format(vmax.shape))
x = x.reshape(b, c, h, w)
res = x - vmax
print('res:\n', res)


# print('vmax:\n', vmax)

exit()



eval_measures_cpu = [random.random() for _ in range(9)]
print('eval_measures_cpu: {}'.format(eval_measures_cpu))
metrices_list = ['silog', 'abs_rel', 'log10', 'rmse', 'sq_rel', 'log_rms', 'delta1', 'delta2', 'delta3']
for i in range(9):
    print('{:>7}: {:7.3f}'.format(metrices_list[i], eval_measures_cpu[i]), end='\n')
exit()


opts = parse_arguments()
config_file_path = './config/MDE_MobileNetV2_LiamEdge.yaml'
opts = load_config_file(config_file_path, opts)

dataloader = BtsDataLoader(opts, 'online_eval')  # train online_eval

for step, sample_batched in enumerate(dataloader.data):
    # print('data loaded.')
    print('step: {}'.format(step))
    # image = torch.autograd.Variable(sample_batched['image'].cuda(non_blocking=True))
    # focal = torch.autograd.Variable(sample_batched['focal'].cuda(non_blocking=True))
    # depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    # exit()

print('dataset is complete.')
exit()


x = torch.arange(1, 13)
y = x.reshape(1, 2, 3, 2)
z = y.repeat(1, 8, 1, 1)
print('y:', y)
print('z:', z)
print('y.shape:{} z.shape:{}'.format(y.shape, z.shape))
exit()



def normalize2img(x: Tensor):
    min_val = x.min()
    max_val = x.max()
    res = ((x - min_val) / (max_val - min_val)) * 255
    res = res.detach().cpu().numpy().astype(np.uint8)
    return res


torch.manual_seed(816)
torch.cuda.manual_seed_all(816)


dataset_loader = datasets.get_dataloader('kitti',
                                       'MobileNetV2Edge',
                                       path='/home/data/glw/hp/datasets/kitti_eigen/test_dataset.zip',
                                       split='test',
                                       augmentation='alhashim',
                                       batch_size=1,
                                       resolution='full',
                                       workers=10)

# NOTE: 加载数据集中的图片
data = next(iter(dataset_loader))
image = data['image']
print('type(image):{}'.format(type(image)))
print('image.shape:{}'.format(image.shape))
print('image max:{} min:{}'.format(torch.max(image), torch.min(image)))

depth = data['depth']
print('type(depth):{}'.format(type(depth)))
print('depth.shape:{}'.format(depth.shape))
print('depth max:{} min:{}'.format(torch.max(depth), torch.min(depth)))

image = image
image = image.numpy().astype(np.uint8)  # [b, h, w, c]

# depth = depth.numpy().astype(np.float32)
depth_np = []
for i in range(depth.shape[0]):
    depth_np.append(normalize2img(depth[i]))
depth_np = np.array(depth_np)
print('depth_np.shape:{}'.format(depth_np.shape))
print('depth_np max:{} min:{}'.format(np.max(depth_np), np.min(depth_np)))
depth = depth_np

cmap_type_list = ["Greys", "plasma", "viridis", "magma"]
cmap_type = cmap_type_list[2]

plt.imshow(image[0])
plt.show()
plt.imshow(depth[0])
plt.show()

exit()


fig = plt.figure(figsize=(12, 9))
for i in range(16):
    if i < 4:
        ax = fig.add_axes([0.02 + 0.245 * i, 0.75, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
        ax.imshow(image[i, :, :, :], cmap_type_list[0])
    elif i < 8:
        ax = fig.add_axes([0.02 + 0.245 * (i - 4), 0.50, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
        ax.imshow(image[i, :, :, :], cmap_type_list[0])
    elif i < 12:
        ax = fig.add_axes([0.02 + 0.245 * (i - 8), 0.25, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
        ax.imshow(depth[i - 8], cmap_type_list[0])
    else:
        ax = fig.add_axes([0.02 + 0.245 * (i - 12), 0.00, 0.245, 0.25])  # 离左边界距离 离下边界距离 宽度比例 高度比例
        ax.imshow(depth[i - 8], cmap_type_list[0])
plt.show()


exit()






class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=2, dilation=2),
        #     # nn.ConvTranspose2d(3, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  #
        #     # nn.Conv2d(3, 24, kernel_size=1, stride=1, padding=0),
        #     # nn.BatchNorm2d(32),
        #     # nn.ReLU(inplace=True),
        #     # nn.AdaptiveAvgPool2d((1, 1)),
        #     # nn.AvgPool2d()
        # )

        # NOTE:
        # input dims: 256 | output dims: 256 * 3 | params (0.196608M)
        # input dims: 256 | output dims: 16 * 3  | params (0.012288M)
        # input dims: 32  | output dims: 16 * 3  | params (0.001536M)
        self.layer2 = nn.Linear(32, 16 * 3, bias=False)

    def forward(self, x):
        return self.layer2(x)


model = Net()
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.6fM" % (total / 1e6))

for _, (key, weight) in enumerate(model.named_parameters()):
    print('key:{} weight shape:{}'.format(key, weight.shape))
    print('----------------')

X = torch.randn(size=(8, 196, 32))
y = model(X)
print('y.shape:{}'.format(y.shape))



exit()

# def extract_zip(input_zip):
#     input_zip=ZipFile(input_zip)
#     return {name: input_zip.read(name) for name in input_zip.namelist()}
#
# data = extract_zip("/home/heping/MyDataset/nyu_test.zip")  # FSIE服务器数据集位置
# rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
# depth = np.load(BytesIO(data['eigen_test_depth.npy']))
# crop = np.load(BytesIO(data['eigen_test_crop.npy']))
#
# print("depth.size:", depth.size)
# print("depth.shape:", depth.shape)


# weights_pth = "./results/best_model.pth"
# state_dict = torch.load(weights_pth, map_location='cpu')
# nums = len(state_dict.keys())
# print(nums)
#
# lst = list(state_dict.keys())
# print(len(lst))
#
# for i in range(nums):
#     print(lst[i])

# print("state_dict:", state_dict.keys())


# up_features = [64, 32, 16]
# model = DualResNet_Backbone(pretrained=True, features=up_features[0])


# x = torch.randn(size=(4, 3, 480, 640))
#
# print(7 * 2 ** 3)  # 56
#
# depths=(1, 2, 8, 2)
# res = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
# print("res:\n", res)

# shutil.copy("./liam_scripts.txt", "./checkpoints/liam_scripts_copy.txt")

# res = np.load("/home/glw/hp/projects/MiT_GUB-2023-04-28/checkpoints/DDRNet-23-slim_pre_GUB_05-16_test/training_metrics.npy")
# print("res:\n", res)

# print(3//4)

# # # 迭代每层来查看输出的形状参数
# # for layer in model:
# #     X = layer(X)
# #     print(layer.__class__.__name__, 'output shape:\t', X.shape)
# features = model(x)
# print("features.size:", features.size())
#
# x_half = F.interpolate(x, scale_factor=.5)
# x_quarter = F.interpolate(x, scale_factor=.25)
#
# print("x_half.size:", x_half.size())
# print("x_quarter.size:", x_quarter.size())


# class ResizeSingle(object):
#     def __init__(self, output_resolution):
#         self.resize = transforms.Resize(output_resolution)
#
#     def __call__(self, x):
#         return self.resize(x)
#
# resize = transforms.Compose([
#     ResizeSingle((240, 320))
# ])

# def resize(x, resolution):
#     tf = transforms.Compose([transforms.Resize(resolution)])
#     return tf(x)
#
# # resize=Compose([Resize((224,224))])
#
# y = resize(x, (10, 10))
# print("y.size():", y.size())


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=2, dilation=2),
#             # nn.ConvTranspose2d(3, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  #
#             # nn.Conv2d(3, 24, kernel_size=1, stride=1, padding=0),
#             # nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             # nn.AdaptiveAvgPool2d((1, 1)),
#             # nn.AvgPool2d()
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# x = torch.randn(size=(4, 3, 480, 640))
# net = Net()
# y = net(x)
# print("y.shape:", y.shape)

# avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
# print("avg_pool.shape:", avg_pool.shape)



# root = "./checkpoints/FasterNet-S_EdgeNetV2_DecoderV7_06-29_2"
# path = os.path.join(root, '*.pth.backup')
# print('path', path)
# all_checkpoints_list = glob.glob(path)
#
# for i in range(len(all_checkpoints_list)):
#     os.rename(all_checkpoints_list[i], all_checkpoints_list[i]+".bp")


# for i in range(0):
#     print('hello')
# exit()



import numpy as np

# # Assuming the given array is called 'arr'
# arr = np.array([5, 5, 8, 1, 5])
#
# arr_unique = np.unique(arr)
# print('arr_unique"{}'.format(arr_unique))
# # Get the indices that would sort the array
# sorted_indices = np.argsort(arr_unique) + 1
# print('sorted_indices"{}'.format(sorted_indices))
#
# sorted_dict = {}
# for i in range(len(arr_unique)):
#     sorted_dict[arr_unique[i]] = sorted_indices[i]
#
# print('sorted_dict"{}'.format(sorted_dict))


rmse = []
mae = []
delta1 = []
delta2 = []
delta3 = []
absrel = []
lg10 = []
gpu_time = []

log_file = "runInfo_FasterNet-S_EdgeNetV3_DecoderV8_07-01.txt"
ues_print = False
with open(os.path.join('.', log_file), 'r') as f:
    for line in f:
        line = line.strip()
        if "RMSE=" in line:
            if ues_print is True:
                print(line)
            rmse.append(float(line[5:]))
        elif "MAE=" in line:
            if ues_print is True:
                print(line)
            mae.append(float(line[4:]))
        elif "Delta1=0." in line:
            if ues_print is True:
                print(line)
            delta1.append(float(line[7:]))
        elif "Delta2=0." in line:
            if ues_print is True:
                print(line)
            delta2.append(float(line[7:]))
        elif "Delta3=0." in line:
            if ues_print is True:
                print(line)
            delta3.append(float(line[7:]))
        elif "REL=" in line:
            if ues_print is True:
                print(line)
            absrel.append(float(line[4:]))
        elif "Lg10=" in line:
            if ues_print is True:
                print(line)
            lg10.append(float(line[5:]))
        elif "t_GPU=0." in line:
            if ues_print is True:
                print(line)
            gpu_time.append(float(line[6:]))

rmse = np.array(rmse)
absrel = np.array(absrel)
lg10 = np.array(lg10)
delta1 = np.array(delta1)
delta2 = np.array(delta2)
delta3 = np.array(delta3)
mae = np.array(mae)
gpu_time = np.array(gpu_time)

# res_list = []
# res_list.append(rmse)
# res_list.append(absrel)
# res_list.append(lg10)
# res_list.append(delta1)
# res_list.append(delta2)
# res_list.append(delta3)

res_dict = {}
res_dict['rmse'] = rmse
res_dict['absrel'] = absrel
res_dict['lg10'] = lg10
res_dict['delta1'] = delta1
res_dict['delta2'] = delta2
res_dict['delta3'] = delta3

sorted_dict = {}
for k, v in res_dict.items():
    arr_unique = np.unique(v)
    # print('arr_unique"{}'.format(arr_unique))
    if 'delta' in k:  # 准确率 需要倒序从大到小排列 最大的排第一 表示效果最好
        sorted_indices = np.argsort(arr_unique)[::-1] + 1
    else:  # 误差项 默认从小到大排列 最小的排第一 表示效果最好
        sorted_indices = np.argsort(arr_unique) + 1
    # print('sorted_indices"{}'.format(sorted_indices))
    sorted_dict[k] = (arr_unique, sorted_indices)

# print('sorted_dict:\n', sorted_dict)

ckpt_dict = {}
for i in range(len(rmse)):  # len(rmse)  有n个ckpt
    temp_list = []
    for k, v in sorted_dict.items():  # 有6个指标
        # print('k:{}----------'.format(k))
        for t in range(len(v[0])):  # 有m个候选值
            # print('res_dict[k][i]:{} v[0][t]:{}'.format(res_dict[k][i], v[0][t]))
            if res_dict[k][i] == v[0][t]:
                # print('v[1][t]:{}'.format(v[1][t]))
                temp_list.append(v[1][t])
    # ckpt_dict['ckpt_' + str(i)] = temp_list
    ckpt_dict['ckpt_'+str(i)] = (temp_list, temp_list.count(1), sum(temp_list)/6)


for k, v in ckpt_dict.items():
    # print('k:{} v:{} nums:{} v_avg:{}'.format(k, v, v.count(1), sum(v) / 6))
    print('k:{} v:{}'.format(k, v))


a = sorted(ckpt_dict.items(), key=lambda x: x[1][2])
for i in range(len(a)):
    print(a[i])

exit()




# Loop through the sorted indices and print the order of each element
for i in range(len(arr)):
    print(f"The element {arr[i]} is at position {np.where(sorted_indices == i)[0][0] + 1} in the sorted array.")

exit()



rmse = []
mae = []
delta1 = []
delta2 = []
delta3 = []
absrel = []
lg10 = []
gpu_time = []

log_file = "runInfo_FasterNet-S_EdgeNetV2_DecoderV7_07-01.txt"
ues_print = False
with open(os.path.join('.', log_file), 'r') as f:
    for line in f:
        line = line.strip()
        if "RMSE=" in line:
            if ues_print is True:
                print(line)
            rmse.append(float(line[5:]))
        elif "MAE=" in line:
            if ues_print is True:
                print(line)
            mae.append(float(line[4:]))
        elif "Delta1=0." in line:
            if ues_print is True:
                print(line)
            delta1.append(float(line[7:]))
        elif "Delta2=0." in line:
            if ues_print is True:
                print(line)
            delta2.append(float(line[7:]))
        elif "Delta3=0." in line:
            if ues_print is True:
                print(line)
            delta3.append(float(line[7:]))
        elif "REL=" in line:
            if ues_print is True:
                print(line)
            absrel.append(float(line[4:]))
        elif "Lg10=" in line:
            if ues_print is True:
                print(line)
            lg10.append(float(line[5:]))
        elif "t_GPU=0." in line:
            if ues_print is True:
                print(line)
            gpu_time.append(float(line[6:]))


rmse = np.array(rmse)
mae = np.array(mae)
delta1 = np.array(delta1)
delta2 = np.array(delta2)
delta3 = np.array(delta3)
absrel = np.array(absrel)
lg10 = np.array(lg10)
gpu_time = np.array(gpu_time)

metrics_dict = {}

rmse_min_loc = np.where(rmse == np.min(rmse))  # rmse_min_loc是一个包含了numpy数组的tuple
# print("rmse_min_loc:{}".format(rmse_min_loc[0]))
metrics_dict["rmse"] = rmse_min_loc[0].tolist()
# print("rmse_min_loc:{}, type:{}".format(rmse_min_loc, type(rmse_min_loc)))

mae_min_loc = np.where(mae == np.min(mae))
# print("mae_min_loc:{}".format(mae_min_loc[0]))
# metrics_dict["mae"] = mae_min_loc[0].tolist()

delta1_max_loc = np.where(delta1 == np.max(delta1))
# print("delta1_max_loc:{}".format(delta1_max_loc[0]))
metrics_dict["delta1"] = delta1_max_loc[0].tolist()

delta2_max_loc = np.where(delta2 == np.max(delta2))
# print("delta2_max_loc:{}".format(delta2_max_loc[0]))
metrics_dict["delta2"] = delta2_max_loc[0].tolist()

delta3_max_loc = np.where(delta3 == np.max(delta3))
# print("delta3_max_loc:{}".format(delta3_max_loc[0]))
metrics_dict["delta3"] = delta3_max_loc[0].tolist()

absrel_min_loc = np.where(absrel == np.min(absrel))
# print("absrel_min_loc:{}".format(absrel_min_loc[0]))
metrics_dict["absrel"] = absrel_min_loc[0].tolist()

lg10_min_loc = np.where(lg10 == np.min(lg10))
# print("lg10_min_loc:{}".format(lg10_min_loc[0]))
metrics_dict["lg10"] = lg10_min_loc[0].tolist()

gpu_time_min_loc = np.where(gpu_time == np.min(gpu_time))
# print("gpu_time_min_loc:{}".format(gpu_time_min_loc[0]))
# metrics_dict["gpu_time"] = gpu_time_min_loc[0].tolist()

for k, v in metrics_dict.items():
    print("k:{}, v:{}".format(k, v))


res_dict = {}  # key: checkpoint序号 value: 对应的metric名字
# target = "rmse"
# for i in range(len(metrics_dict["rmse"])):
#     candidate = metrics_dict["rmse"][i]
#     res_dict[candidate] = ["rmse"]
#     # print("res_dict:{}".format(res_dict))
#
#     # temp_dict = metrics_dict.copy()
#     temp_dict = copy.deepcopy(metrics_dict)  # 深拷贝
#     temp_dict.pop("rmse")
#
#     for k, v in temp_dict.items():
#         for j in range(len(v)):
#             if candidate == v[j]:
#                 res_dict[candidate].append(k)
#                 metrics_dict[k].remove(v[j])


# target_list = ["rmse", "mae", "delta1", "delta2", "delta3", "absrel", "lg10", "gpu_time"]
target_list = ["rmse", "delta1", "delta2", "delta3", "absrel", "lg10"]
for target in target_list:
    for i in range(len(metrics_dict[target])):
        candidate = metrics_dict[target][i]
        res_dict[candidate] = [target]

        temp_dict = copy.deepcopy(metrics_dict)  # 深拷贝
        temp_dict.pop(target)

        for k, v in temp_dict.items():
            for j in range(len(v)):
                if candidate == v[j]:
                    res_dict[candidate].append(k)
                    metrics_dict[k].remove(v[j])

# for k, v in metrics_dict.items():
#     print("metrics_dict k:{}, v:{}".format(k, v))

for k, v in res_dict.items():
    # print("res_dict k:{}, v:{}".format(k, v))
    print("ckpt_{} counts:{} metrics:{}".format(k, len(v), v))


# for i in range(len(rmse)):




exit()




root = "./checkpoints/FasterNet-X-V2_EdgeNetV1_4L_DecoderV6_4L_06-29"
path = os.path.join(root, '*.pth')
print('path', path)
all_checkpoints_list = glob.glob(path)

# ckp_list = [27, 35, 36, 37, 38, 20, 32, 29]
# ckp_list = [25, 24, 14, 20, 27, 31, 30, 33, 34, 35]
# ckp_list = [12, 23, 24, 27, 14, 16, 25, 28]
ckp_list = [20, 25, 26, 27, 28]

for i in range(len(ckp_list)):
    preserved = os.path.join(root, 'checkpoint_{}.pth'.format(ckp_list[i]))
    print("preserved:", preserved)
    all_checkpoints_list.remove(preserved)

for i in range(len(all_checkpoints_list)):
    print('deleting:{}'.format(all_checkpoints_list[i]))
    os.remove(all_checkpoints_list[i])
