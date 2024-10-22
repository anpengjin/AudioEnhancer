import warnings
from collections import defaultdict
from typing import Dict, Final, Iterable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function

from net.deepfilternet.modules import LocalSnrTarget


def as_complex(x: Tensor):
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dimension need to be of length 2 (re + im), but got {x.shape}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)

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

