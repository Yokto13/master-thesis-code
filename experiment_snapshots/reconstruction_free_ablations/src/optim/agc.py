# MIT License

# Copyright (c) 2026 Naoki Morihira

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections.abc import Iterable
from typing import Union

import torch
from torch import Tensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_grad_agc_(parameters: _tensor_or_tensors, clip: float = 0.3, pmin: float = 1e-3, foreach: bool | None = None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    params = []
    grads = []
    for p in parameters:
        if p.grad is not None:
            params.append(p)
            grads.append(p.grad)

    if len(grads) == 0:
        return
    grouped: dict[tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]] = (
        _group_tensors_by_device_and_dtype([params, grads])
    )  # type: ignore[assignment]

    for (device, _), ([device_params, device_grads], _) in grouped.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            pnorm = torch._foreach_norm(device_params, ord=2)
            gnorm = torch._foreach_norm(device_grads, ord=2)
            upper = torch._foreach_mul(torch._foreach_maximum(pnorm, pmin), clip)
            scale = torch._foreach_reciprocal(torch._foreach_maximum(torch._foreach_div(gnorm, upper), 1.0))
            torch._foreach_mul_(device_grads, scale)
        elif not foreach:
            for p, g in zip(device_params, device_grads):
                pnorm = torch.norm(p, p=2)
                gnorm = torch.norm(g, p=2)
                upper = torch.tensor(clip) * torch.maximum(torch.tensor(pmin), pnorm)
                scale = 1 / torch.maximum(torch.tensor(1.0), gnorm / upper)
                g.detach().mul_(scale)
        else:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
