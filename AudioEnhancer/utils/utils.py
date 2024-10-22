#encoding:utf-8
import os
import yaml

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function

def load_yaml(yaml_path):
    # 打开yaml文件
    with open(yaml_path, 'r', encoding='utf-8') as file:
        # 将yaml文件内容转换为字典或列表格式
        data = yaml.safe_load(file)
    return data


class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))
