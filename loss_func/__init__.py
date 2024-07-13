import importlib
import os

import torch
from torch import nn, Tensor
# from .mde_loss.L1 import L1Loss
# from .mde_loss.SSIM import SSIMLoss

from utils import logger


SUPPORTED_CLS_LOSS_FNS = []
MDE_LOSS_FN_REGISTRY = {}


def register_mde_loss_fn(name):
    def register_fn(cls):
        if name in SUPPORTED_CLS_LOSS_FNS:
            raise ValueError(
                "Cannot register duplicate mde loss function ({})".format(
                    name
                )
            )

        MDE_LOSS_FN_REGISTRY[name] = cls
        SUPPORTED_CLS_LOSS_FNS.append(name)
        return cls

    return register_fn


def get_mde_loss(opts, *args, **kwargs):
    loss_fn_name_list = getattr(opts, "loss.name", ["L1", "ssim"])
    loss_fn_list = list()

    for i in range(len(loss_fn_name_list)):
        if loss_fn_name_list[i] in SUPPORTED_CLS_LOSS_FNS:
            loss_fn_name = loss_fn_name_list[i]
            loss_fn_list.append(MDE_LOSS_FN_REGISTRY[loss_fn_name](opts, *args, **kwargs))

    return loss_fn_list


class LossCriterion(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()

        self.criteria = get_mde_loss(opts=opts, *args, **kwargs)  # 包含loss函数实体的列表
        logger.info("MDELoss self.criteria:{}".format(self.criteria))

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        num_loss = len(self.criteria)
        loss = torch.tensor([0.0], dtype=torch.float, requires_grad=True).to(device=prediction.device)  # 数据类型torch.float32

        for i in range(num_loss):
            if self.criteria[i].__class__.__name__ == "L1":
                loss += 0.1 * self.criteria[i](prediction=prediction, target=target)
            elif self.criteria[i].__class__.__name__ == "SSIM":
                loss += 1.0 * torch.clamp(
                    (1 - self.criteria[i](prediction=prediction, target=target, val_range=1000.0 / 10.0)) * 0.5, 0, 1)

        return loss

    def __repr__(self):
        return self.criteria.__repr__()


# NOTE: automatically import different loss functions
loss_fn_dir = os.path.dirname(__file__)
for file in os.listdir(loss_fn_dir):
    path = os.path.join(loss_fn_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "loss_func." + model_name
        )


