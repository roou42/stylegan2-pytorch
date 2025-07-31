import math
import random
import torch
from torch import nn
from torch.nn import functional as F

from upfirdn2d import upfirdn2d  

#LeakyReLU wrapper
class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1)) if bias else None
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, input):
        if self.bias is not None:
            return self.activation(input + self.bias)
        return self.activation(input)


def fused_leaky_relu(input, bias):
    return F.leaky_relu(input + bias.view(1, -1), negative_slope=0.2)


class Conv2dGradFix:
    @staticmethod
    def conv2d(input, weight, bias=None, stride=1, padding=0, groups=1):
        return F.conv2d(input, weight, bias, stride, padding, groups)

    @staticmethod
    def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, groups=1):
        return F.conv_transpose2d(input, weight, bias, stride, padding, groups)

conv2d_gradfix = Conv2dGradFix
