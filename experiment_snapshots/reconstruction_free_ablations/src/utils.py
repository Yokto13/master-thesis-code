import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from ale_py import ALEState
from gymnasium import Wrapper


def unimix(logits, unimix_ratio):
    probs = torch.softmax(logits, dim=-1)
    p = (1.0 - unimix_ratio) * probs + unimix_ratio / probs.shape[-1]
    return p


def symlog(x):
    """
    symlog(x) = sign(x) * ln(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def extract_env_state(env) -> ALEState:
    """Extracts ale state from gym environment"""
    ale = env.unwrapped.ale
    return ale.cloneState()


def set_env_state(env, state) -> Wrapper:
    """Sets ale state for gym environment"""
    env.unwrapped.ale.restoreState(state)
    return env


@torch.no_grad()
def trunc_normal_init(tensor, fan="in", scale=1.0, fan_shape=None):
    """
    In-place truncated normal initialization matching DreamerV3's JAX implementation.

    Args:
        tensor: PyTorch tensor to initialize
        fan: Fan mode - 'in', 'out', 'avg', or 'none'
        scale: Additional scaling factor (default 1.0)
        fan_shape: Optional (fan_in, fan_out) tuple to override fan computation.
                   Useful when the tensor layout doesn't match standard conventions
                   (e.g. BlockLinear kernel shape (G, I/G, O/G)).
    """
    if fan_shape is not None:
        fan_in, fan_out = fan_shape
    else:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    # Select fan based on mode
    fan_dict = {"avg": (fan_in + fan_out) / 2, "in": fan_in, "out": fan_out, "none": 1}
    fan_value = fan_dict[fan]

    # Truncated normal with std=1, truncated at -2 and 2
    nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)

    # Apply scaling: 1.1368 is the correction factor for truncated normal
    # and sqrt(1/fan) is the fan-based scaling
    tensor.mul_(1.1368 * math.sqrt(1.0 / fan_value) * scale)


def dict_apply(func: Callable, d: dict):
    """Recursively applies a function to all non-dict values in a nested dictionary."""
    return {k: dict_apply(func, v) if isinstance(v, dict) else func(v) for k, v in d.items()}


def get_post_burn_in(burn_in_steps: int, x: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """Returns the tensor after burn-in steps along the specified axis (default is dim=1 for time)."""
    slices = [slice(None)] * x.dim()
    slices[axis] = slice(burn_in_steps, None)
    return x[tuple(slices)]


def random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    """DrQ-style random-shift augmentation with edge (replicate) padding.

    Pads the image by ``pad`` pixels on every side and crops back a random window
    of the original size. A single shift is drawn per sequence and shared across
    time and channels, so the temporal dynamics are preserved -- drawing an
    independent shift per frame would inject unpredictable motion that the
    forward-prediction (``pred``) loss cannot model.

    Args:
        x: image tensor of shape (B, T, C, H, W); cast to float internally.
        pad: padding/shift magnitude in pixels (e.g. 4).

    Returns:
        Float tensor of shape (B, T, C, H, W), same value range as the input.
    """
    b, t, c, h, w = x.shape
    x = x.float()
    padded = F.pad(x.reshape(b * t, c, h, w), (pad, pad, pad, pad), mode="replicate")
    padded = padded.reshape(b, t, c, h + 2 * pad, w + 2 * pad)

    out = torch.empty_like(x)
    for i in range(b):
        # CPU-sampled python ints (seeded by the global RNG) -> no device sync.
        oh = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        ow = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        out[i] = padded[i, :, :, oh : oh + h, ow : ow + w]
    return out


def get_optimizer_param_groups(model, exclude_keywords=["in", "out", "head", "dyn_to_gru"]):
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        should_exclude = any(keyword in name for keyword in exclude_keywords)

        if param.ndim >= 2 and not should_exclude:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params
