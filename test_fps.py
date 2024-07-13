import torch
import time
from typing import Tuple


def pyTorch_speedtest(model, resolution: Tuple = (480, 640), num_test_runs=200, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model.eval()
    torch.cuda.empty_cache()
    times = 0.0
    warm_up_runs = 10
    for i in range(num_test_runs + warm_up_runs):
        if i == warm_up_runs:
            times = 0.0

        x = torch.randn([1, 3, *resolution]).cuda()
        torch.cuda.synchronize()  # Synchronize transfer to cuda

        t0 = time.time()
        _ = model(x)
        torch.cuda.synchronize()
        times += time.time() - t0

    times = times / num_test_runs
    fps = 1 / times
    print('[PyTorch] Runtime: {:.6}s'.format(times))
    print('[PyTorch] Runtime: {:.3}ms'.format(times * 1000))
    print('[PyTorch] FPS: {:.2f}'.format(fps))
    return times, fps


def tensorRT_speedtest(self, num_test_runs=200):
    """
    类中的函数吗 我还没有修改 上面那个是我一直在用的
    Args:
        self:
        num_test_runs:

    Returns:

    """
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


if __name__ == '__main__':

    model = None

    time_list = []
    fps_list = []
    for i in range(5):
        times, fps = pyTorch_speedtest(model, num_test_runs=200, seed=i)
        # print('times:{:.6f} fps:{:.2f}'.format(times, fps))
        time_list.append(times)
        fps_list.append(fps)

    print('average times:{:.6f}s'.format(sum(time_list) / len(time_list)))
    print('average times:{:.3f}ms'.format(1000 * sum(time_list) / len(time_list)))
    print('average fps:{:.2f}'.format(sum(fps_list) / len(fps_list)))
