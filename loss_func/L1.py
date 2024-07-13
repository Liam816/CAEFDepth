from torch import nn
from torch import Tensor
from loss_func import register_mde_loss_fn


@register_mde_loss_fn(name="L1")
class L1Loss(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()

    def forward(self, prediction: Tensor, target: Tensor, *args, **kwargs):
        return nn.L1Loss()(prediction, target)



