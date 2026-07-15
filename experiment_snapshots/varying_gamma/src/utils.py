import math
from typing import Callable

import torch
import torch.nn as nn
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


def exponential_decay_scheduler(
    decay_period: float,
    initial_value: float,
    final_value: float,
    reverse: bool = False,
) -> Callable[[int], float]:
    """Based on BBF's JAX implementation"""
    if reverse:
        initial_value = 1.0 - initial_value
        final_value = 1.0 - final_value

    start = math.log(initial_value)
    end = math.log(final_value)

    if decay_period == 0:
        return lambda step: final_value

    def scheduler(step: int) -> float:
        bonus = max(0.0, min(1.0, (decay_period - step) / decay_period))
        value = math.exp(bonus * (start - end) + end)
        if reverse:
            value = 1.0 - value
        return value

    return scheduler
