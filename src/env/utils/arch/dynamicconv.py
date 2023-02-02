"""
28-Sep-21
https://github.com/TArdelean/DynamicConvolution/blob/master/dynamic_convolutions.py
https://github.com/TArdelean/DynamicConvolution/blob/master/models/common.py
"""
from collections.abc import Iterable
import itertools

import torch
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch import nn

import torch
from torch import nn
from torch.nn import *
from collections import OrderedDict
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    TYPE_CHECKING,
    overload,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T", bound=Module)


class Conv2dWrapper(nn.Conv2d):
    """
    Wrapper for pytorch Conv2d class which can take additional parameters(like temperature) and ignores them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, temperature) -> torch.Tensor:
        return x


class BaseModel(TempModule):
    def __init__(self, ConvLayer):
        super().__init__()
        self.ConvLayer = ConvLayer


class TemperatureScheduler:
    def __init__(self, initial_value, final_value=None, final_epoch=None):
        self.initial_value = initial_value
        self.final_value = final_value if final_value else initial_value
        self.final_epoch = final_epoch if final_epoch else 1
        self.step = (
            0
            if self.final_epoch == 1
            else (final_value - initial_value) / (final_epoch - 1)
        )

    def get(self, crt_epoch=None):
        crt_epoch = crt_epoch if crt_epoch else self.final_epoch
        return self.initial_value + (min(crt_epoch, self.final_epoch) - 1) * self.step


class CustomSequential(TempModule):
    """Sequential container that supports passing temperature to TempModule"""

    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, x, temperature):
        for layer in self.layers:
            if isinstance(layer, TempModule):
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x

    def __getitem__(self, idx):
        return CustomSequential(*list(self.layers[idx]))
        # if isinstance(idx, slice):
        #     return self.__class__(OrderedDict(list(self.layers.items())[idx]))
        # else:
        #     return self._get_item_by_idx(self.layers.values(), idx)


# Implementation inspired from
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py#L38 and
# https://github.com/pytorch/pytorch/issues/7455
class SmoothNLLLoss(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super().__init__()
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, prediction, target):
        with torch.no_grad():
            smooth_target = torch.zeros_like(prediction)
            n_class = prediction.size(self.dim)
            smooth_target.fill_(self.smoothing / (n_class - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-smooth_target * prediction, dim=self.dim))


class AttentionLayer(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super().__init__()
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.to_scores = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, nof_kernels),
        )

    def forward(self, x, temperature=1):
        out = self.global_pooling(x)
        scores = self.to_scores(out)
        return F.softmax(scores / temperature, dim=-1)


class DynamicConvolution(TempModule):
    def __init__(
        self,
        nof_kernels,
        reduce,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        """
        Implementation of Dynamic convolution layer
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param kernel_size: size of the kernel.
        :param groups: controls the connections between inputs and outputs.
        in_channels and out_channels must both be divisible by groups.
        :param nof_kernels: number of kernels to use.
        :param reduce: Refers to the size of the hidden layer in attention: hidden = in_channels // reduce
        :param bias: If True, convolutions also have a learnable bias
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groups = groups
        self.conv_args = {"stride": stride, "padding": padding, "dilation": dilation}
        self.nof_kernels = nof_kernels
        self.attention = AttentionLayer(
            in_channels, max(1, in_channels // reduce), nof_kernels
        )
        self.kernel_size = _pair(kernel_size)
        self.kernels_weights = nn.Parameter(
            torch.Tensor(
                nof_kernels, out_channels, in_channels // self.groups, *self.kernel_size
            ),
            requires_grad=True,
        )
        if bias:
            self.kernels_bias = nn.Parameter(
                torch.Tensor(nof_kernels, out_channels), requires_grad=True
            )
        else:
            self.register_parameter("kernels_bias", None)
        self.initialize_parameters()

    def initialize_parameters(self):
        for i_kernel in range(self.nof_kernels):
            init.kaiming_uniform_(self.kernels_weights[i_kernel], a=math.sqrt(5))
        if self.kernels_bias is not None:
            bound = 1 / math.sqrt(self.kernels_weights[0, 0].numel())
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    def forward(self, x, temperature=1):
        batch_size = 1
        magic = 4

        alphas = self.attention(x, temperature)

        agg_weights = torch.sum(
            torch.mul(
                self.kernels_weights.unsqueeze(0),
                alphas.view(batch_size, 4, 1, 1, 1, 1),
            ),
            dim=1,
        )
        # Group the weights for each batch to conv2 all at once
        agg_weights = agg_weights.view(
            -1, *agg_weights.shape[-3:]
        )  # batch_size*out_c X in_c X kernel_size X kernel_size
        if self.kernels_bias is not None:
            agg_bias = torch.sum(
                torch.mul(
                    self.kernels_bias.unsqueeze(0), alphas.view(batch_size, 4, 1) # 4
                ),
                dim=1,
            )
            agg_bias = agg_bias.view(192)
        else:
            agg_bias = None
        
        #x_grouped = x.view(1, -1, *x.shape[-2:])  # 1 X batch_size*out_c X H X W
        #x_grouped = x

        out = F.conv2d(
            x,
            agg_weights,
            agg_bias,
            groups=self.groups * batch_size,
            **self.conv_args
        )  # 1 X batch_size*out_C X H' x W'
        #out = out.view(batch_size, -1, *out.shape[-2:])  # batch_size X out_C X H' x W'
        return out


class FlexibleKernelsDynamicConvolution:
    def __init__(self, Base, nof_kernels, reduce):
        if isinstance(nof_kernels, Iterable):
            self.nof_kernels_it = iter(nof_kernels)
        else:
            self.nof_kernels_it = itertools.cycle([nof_kernels])
        self.Base = Base
        self.reduce = reduce

    def __call__(self, *args, **kwargs):
        return self.Base(next(self.nof_kernels_it), self.reduce, *args, **kwargs)


def dynamic_convolution_generator(nof_kernels, reduce):
    return FlexibleKernelsDynamicConvolution(DynamicConvolution, nof_kernels, reduce)


if __name__ == "__main__":
    torch.manual_seed(41)
    t = torch.randn(1, 3, 16, 16)
    conv = DynamicConvolution(
        3, 1, in_channels=3, out_channels=8, kernel_size=3, padding=1, bias=True
    )
