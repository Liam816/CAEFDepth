import time
import os
import argparse

import torch
import torchvision
import tensorrt as trt
import torch2trt
import torch_tensorrt
print('all tensorrt relevant components are ready.')
exit()

import matplotlib.pyplot as plt

from data import datasets
from model import loader
from metrics import AverageMeter, Result
from data import MyTransforms

from utils.main_utils import parse_arguments, load_config_file
from utils import logger

max_depths = {
    'kitti': 80.0,
    'nyu': 10.0,
    'nyu_reduced': 10.0,
}
nyu_res = {
    'full': (480, 640),
    'half': (240, 320),
    'mini': (224, 224)}
kitti_res = {
    'full': (384, 1280),
    'half': (192, 640)}
resolutions = {
    'nyu': nyu_res,
    'nyu_reduced': nyu_res,
    'kitti': kitti_res}
crops = {
    'kitti': [128, 381, 45, 1196],
    'nyu': [20, 460, 24, 616],
    'nyu_reduced': [20, 460, 24, 616]}


class Inference_Engine():
    def __init__(self, opts):
        dataset_name = getattr(opts, "dataset.name", "nyu_reduced")
        resolution_opt = getattr(opts, "common.resolution", "full")
        model_name = getattr(opts, "model.name", "GuideDepth")

        self.maxDepth = max_depths[dataset_name]
        self.res_dict = resolutions[dataset_name]
        self.resolution = self.res_dict[resolution_opt]
        self.resolution_keyword = resolution_opt
        print('Resolution for Eval: {}'.format(self.resolution))
        print('Maximum Depth of Dataset: {}'.format(self.maxDepth))
        self.crop = crops[dataset_name]

        checkpoint_path = getattr(args, "common.save_checkpoint", "./checkpoints")
        inference_results_path = os.path.join(checkpoint_path, "inference_results")

        self.result_dir = inference_results_path
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.results_filename = '{}_{}_{}'.format(dataset_name,
                                                  resolution_opt,
                                                  model_name)

        cuda_visible = getattr(args, "common.cuda_visible", '0')
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info("cuda_visible:{}".format(cuda_visible))
        logger.info("self.device:{}".format(self.device))

        self.model = loader.load_model(opts)
        # self.model = self.model.eval().cuda()
        self.model = self.model.eval().to(self.device)

        inference_path = getattr(args, "dataset.root_inference", None)
        num_workers = getattr(args, "common.num_workers", 4)
        eval_mode = getattr(args, "common.eval_mode", "alhashim")
        logger.info("inference_path:{}".format(inference_path))
        logger.info("num_workers:{}".format(num_workers))
        logger.info("model_name:{}".format(model_name))

        # 计算模型的参数量
        total = sum([param.nelement() for param in self.model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

        self.test_loader = datasets.get_dataloader(dataset_name,
                                                   model_name,
                                                   path=inference_path,
                                                   split='test',
                                                   batch_size=1,
                                                   augmentation=eval_mode,
                                                   resolution=resolution_opt,
                                                   uncompressed=True,  # 使用的是NYU_Testset文件夹中的数据
                                                   workers=num_workers)

        if resolution_opt == 'half':
            self.upscale_depth = torchvision.transforms.Resize(self.res_dict['full'])  # To Full res
            self.downscale_image = torchvision.transforms.Resize(self.resolution)  # To Half res

        self.to_tensor = MyTransforms.ToTensor(test=True, maxDepth=self.maxDepth)

        self.visualize_images = []

        # self.trt_model, _ = self.convert_PyTorch_to_TensorRT()
        # self.convert_PyTorch_to_TensorRT_liam()

        self.run_evaluation()

    def run_evaluation(self):
        speed_pyTorch = self.pyTorch_speedtest_raw()
        # speed_tensorRT = self.tensorRT_speedtest_raw()
        # # self.tensorRT_speedtest_liam()
        #
        # average = self.tensorRT_evaluate()
        # self.save_results(average, speed_tensorRT, speed_pyTorch)

    def pyTorch_speedtest_raw(self, num_test_runs=200):
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        for i in range(num_test_runs + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            x = torch.randn([1, 3, *self.resolution]).cuda()
            torch.cuda.synchronize()  # Synchronize transfer to cuda

            t0 = time.time()
            result = self.model(x)
            torch.cuda.synchronize()
            times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times
        print('[PyTorch] Runtime: {}s'.format(times))
        print('[PyTorch] FPS: {}\n'.format(fps))
        return times

    def tensorRT_speedtest_raw(self, num_test_runs=200):
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        for i in range(num_test_runs + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            x = torch.randn([1, 3, *self.resolution]).cuda()
            torch.cuda.synchronize()  # Synchronize transfer to cuda

            t0 = time.time()
            result = self.trt_model(x)
            torch.cuda.synchronize()
            times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times
        print('[tensorRT] Runtime: {}s'.format(times))
        print('[tensorRT] FPS: {}\n'.format(fps))
        return times

    def pyTorch_speedtest(self, num_test_runs=200):
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        for i in range(num_test_runs + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            x = torch.randn([1, 3, *self.resolution]).cuda()
            torch.cuda.synchronize()  # Synchronize transfer to cuda

            t0 = time.time()
            result = self.model(x)
            torch.cuda.synchronize()
            times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times
        print('[PyTorch] Runtime: {}s'.format(times))
        print('[PyTorch] FPS: {}\n'.format(fps))
        return times

    def tensorRT_speedtest(self, num_test_runs=200):
        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        for i in range(num_test_runs + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            x = torch.randn([1, 3, *self.resolution]).cuda()
            torch.cuda.synchronize()  # Synchronize transfer to cuda

            t0 = time.time()
            result = self.trt_model(x)
            torch.cuda.synchronize()
            times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times
        print('[tensorRT] Runtime: {}s'.format(times))
        print('[tensorRT] FPS: {}\n'.format(fps))
        return times

    def tensorRT_speedtest_liam(self, num_test_runs=200):

        model_trt = torch2trt.TRTModule()
        model_path = os.path.join(self.result_dir, "{}_8218.pth".format(self.results_filename))
        engine_path = os.path.join(self.result_dir, '{}.engine'.format(self.results_filename))
        model_trt.load_state_dict(torch.load(model_path))
        print("model_trt weights loaded.")

        torch.cuda.empty_cache()
        times = 0.0
        warm_up_runs = 10
        for i in range(num_test_runs + warm_up_runs):
            if i == warm_up_runs:
                times = 0.0

            x = torch.randn([1, 3, *self.resolution]).cuda()
            torch.cuda.synchronize()  # Synchronize transfer to cuda

            t0 = time.time()
            result = model_trt(x)
            torch.cuda.synchronize()
            times += time.time() - t0

        times = times / num_test_runs
        fps = 1 / times
        print('[tensorRT LIAM] Runtime: {}s'.format(times))
        print('[tensorRT LIAM] FPS: {}\n'.format(fps))
        return times

    # def tensorRT_speedtest(self, num_test_runs=200):
    #     from model.GuideDepth import GuideDepth
    #     from nets.models.fasternet_PCGUB import FasterNetPCGUB
    #
    #     model_name = getattr(opts, "model.name", "GuideDepth")
    #     if model_name == 'DDRNetGUB':
    #         trt_model = GuideDepth(True)
    #     elif model_name == "FasterNetPCGUB":
    #         trt_model = FasterNetPCGUB(opts)
    #     else:
    #         logger.error("{} is not supported yet.".format(model_name))
    #
    #     model_weight_path = os.path.join(self.result_dir, "{}.pth".format(self.results_filename))
    #     weights = torch.load(model_weight_path)
    #     trt_model.load_state_dict(weights)
    #
    #     torch.cuda.empty_cache()
    #     times = 0.0
    #     warm_up_runs = 10
    #     for i in range(num_test_runs + warm_up_runs):
    #         if i == warm_up_runs:
    #             times = 0.0
    #
    #         x = torch.randn([1, 3, *self.resolution]).cuda()
    #         torch.cuda.synchronize()  # Synchronize transfer to cuda
    #
    #         t0 = time.time()
    #         result = trt_model(x)
    #         torch.cuda.synchronize()
    #         times += time.time() - t0
    #
    #     times = times / num_test_runs
    #     fps = 1 / times
    #     print('[tensorRT] Runtime: {}s'.format(times))
    #     print('[tensorRT] FPS: {}\n'.format(fps))
    #     return times

    # def tensorRT_speedtest(self, num_test_runs=200):
    #
    #     # 加载runtime，记录log
    #     runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    #     file_path = os.path.join(self.result_dir, '{}.engine'.format(self.results_filename))
    #     # 反序列化模型
    #     engine = runtime.deserialize_cuda_engine(open(file_path, "rb").read())
    #
    #     import pycuda.driver as cuda
    #     h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    #     h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    #     # Allocate device memory for inputs and outputs.
    #     d_input = cuda.mem_alloc(h_input.nbytes)
    #     d_output = cuda.mem_alloc(h_output.nbytes)
    #
    #     stream = cuda.Stream()
    #
    #     context = engine.create_execution_context()
    #     return context, h_input, h_output, stream, d_input, d_output
    #
    #
    #     torch.cuda.empty_cache()
    #     times = 0.0
    #     warm_up_runs = 10
    #     for i in range(num_test_runs + warm_up_runs):
    #         if i == warm_up_runs:
    #             times = 0.0
    #
    #         x = torch.randn([1, 3, *self.resolution]).cuda()
    #         torch.cuda.synchronize()  # Synchronize transfer to cuda
    #
    #         t0 = time.time()
    #         result = self.trt_model(x)
    #         torch.cuda.synchronize()
    #         times += time.time() - t0
    #
    #     times = times / num_test_runs
    #     fps = 1 / times
    #     print('[tensorRT] Runtime: {}s'.format(times))
    #     print('[tensorRT] FPS: {}\n'.format(fps))
    #     return times

    def convert_PyTorch_to_TensorRT(self):
        x = torch.ones([1, 3, *self.resolution]).cuda()
        print('[tensorRT] Starting TensorRT conversion')
        model_trt = torch2trt.torch2trt(self.model, [x], fp16_mode=True)
        print("[tensorRT] Model converted to TensorRT")

        model_path = os.path.join(self.result_dir, "{}_8218.pth".format(self.results_filename))
        torch.save(model_trt.state_dict(), model_path)
        print("[tensorRT] Model saved")

        TRT_LOGGER = trt.Logger()
        file_path = os.path.join(self.result_dir, '{}_8218.engine'.format(self.results_filename))
        with open(file_path, 'wb') as f:
            f.write(model_trt.engine.serialize())

        with open(file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        print('[tensorRT] Engine serialized\n')
        return model_trt, engine

    def convert_PyTorch_to_TensorRT_liam(self):
        x = torch.ones([1, 3, *self.resolution]).cuda()
        print('[tensorRT] Starting TensorRT conversion')

        inputs = [
            torch_tensorrt.Input(
                min_shape=[1, 1, 120, 160],
                opt_shape=[1, 1, 240, 320],
                max_shape=[1, 1, 480, 640],
                dtype=torch.half,
            )
        ]
        enabled_precisions = {torch.float, torch.half}  # Run with fp16

        print('[torch_tensorrt] compiling...\n')
        trt_ts_module = torch_tensorrt.compile(
            self.model, inputs=inputs, enabled_precisions=enabled_precisions
        )
        print('[torch_tensorrt] compiled\n')

        trt_ts_module_path = os.path.join(self.result_dir, "{}.ts".format(self.results_filename))
        torch.jit.save(trt_ts_module, trt_ts_module_path)
        print('[torch_tensorrt] trt_ts_module saved\n')

    def tensorRT_evaluate(self):
        torch.cuda.empty_cache()
        self.model = None
        average_meter = AverageMeter()

        dataset = self.test_loader.dataset
        for i, data in enumerate(dataset):
            t0 = time.time()
            image, gt = data
            packed_data = {'image': image, 'depth': gt}
            data = self.to_tensor(packed_data)
            image, gt = self.unpack_and_move(data)
            image = image.unsqueeze(0)
            gt = gt.unsqueeze(0)

            image_flip = torch.flip(image, [3])
            gt_flip = torch.flip(gt, [3])
            if self.resolution_keyword == 'half':
                image = self.downscale_image(image)
                image_flip = self.downscale_image(image_flip)

            torch.cuda.synchronize()
            data_time = time.time() - t0

            t0 = time.time()
            inv_prediction = self.trt_model(image)
            prediction = self.inverse_depth_norm(inv_prediction)
            torch.cuda.synchronize()
            gpu_time0 = time.time() - t0

            t1 = time.time()
            inv_prediction_flip = self.trt_model(image_flip)
            prediction_flip = self.inverse_depth_norm(inv_prediction_flip)
            torch.cuda.synchronize()
            gpu_time1 = time.time() - t1

            if self.resolution_keyword == 'half':
                prediction = self.upscale_depth(prediction)
                prediction_flip = self.upscale_depth(prediction_flip)

            if i in self.visualize_images:
                self.save_image_results(image, gt, prediction, i)

            gt = gt[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
            gt_flip = gt_flip[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
            prediction = prediction[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
            prediction_flip = prediction_flip[:, :, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]

            result = Result()
            result.evaluate(prediction.data, gt.data)
            average_meter.update(result, gpu_time0, data_time, image.size(0))

            result_flip = Result()
            result_flip.evaluate(prediction_flip.data, gt_flip.data)
            average_meter.update(result_flip, gpu_time1, data_time, image.size(0))

        # Report
        avg = average_meter.average()
        current_time = time.strftime('%H:%M', time.localtime())
        print('\n*\n'
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'Delta1={average.delta1:.3f}\n'
              'Delta2={average.delta2:.3f}\n'
              'Delta3={average.delta3:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'Lg10={average.lg10:.3f}\n'
              't_GPU={time:.3f}\n'.format(
            average=avg, time=avg.gpu_time))
        return avg

    def save_results(self, average, trt_speed, pyTorch_speed):
        file_path = os.path.join(self.result_dir, '{}.txt'.format(self.results_filename))
        with open(file_path, 'w') as f:
            f.write('s[PyTorch], s[tensorRT], RMSE,MAE,REL,Lg10,Delta1,Delta2,Delta3\n')
            f.write('{pyTorch_speed:.3f}'
                    ',{trt_speed:.3f}'
                    ',{average.rmse:.3f}'
                    ',{average.mae:.3f}'
                    ',{average.absrel:.3f}'
                    ',{average.lg10:.3f}'
                    ',{average.delta1:.3f}'
                    ',{average.delta2:.3f}'
                    ',{average.delta3:.3f}'.format(
                average=average, trt_speed=trt_speed, pyTorch_speed=pyTorch_speed))

    def inverse_depth_norm(self, depth):
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        return depth

    def depth_norm(self, depth):
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        return depth

    def unpack_and_move(self, data):
        if isinstance(data, (tuple, list)):
            image = data[0].to(self.device, non_blocking=True)
            gt = data[1].to(self.device, non_blocking=True)
            return image, gt
        if isinstance(data, dict):
            keys = data.keys()
            image = data['image'].to(self.device, non_blocking=True)
            gt = data['depth'].to(self.device, non_blocking=True)
            return image, gt
        print('Type not supported')

    def save_image_results(self, image, gt, prediction, image_id):
        img = image[0].permute(1, 2, 0).cpu()
        gt = gt[0, 0].permute(0, 1).cpu()
        prediction = prediction[0, 0].permute(0, 1).detach().cpu()
        error_map = gt - prediction
        vmax_error = self.maxDepth / 10.0
        vmin_error = 0.0
        cmap = 'viridis'

        vmax = torch.max(gt[gt != 0.0])
        vmin = torch.min(gt[gt != 0.0])

        save_to_dir = os.path.join(self.result_dir, 'image_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'errors_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        errors = ax.imshow(error_map, vmin=vmin_error, vmax=vmax_error, cmap='Reds')
        fig.colorbar(errors, ax=ax, shrink=0.8)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'gt_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(gt, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'depth_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(prediction, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.savefig(save_to_dir)
        plt.clf()


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


if __name__ == '__main__':
    # print("hello world")
    # print("trt.__version__:", trt.__version__)
    # # print("torch2trt.__version__:", torch2trt.__version__)
    # exit()

    args = parse_arguments()
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
    # config_file_path = "./config/MDE_MobileViTv1-s_LiamEdge.yaml"
    # config_file_path = "./config/MDE_FasterNet-X_LiamEdge.yaml"
    opts = load_config_file(config_file_path, args)

    engine = Inference_Engine(args)
