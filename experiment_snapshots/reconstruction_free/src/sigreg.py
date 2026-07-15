"""Minimal vendored SIGReg implementation.

Adapted from https://github.com/galilai-group/lejepa/tree/main/lejepa
DDP / all_reduce machinery stripped (single-process only).
"""

import torch
import torch.nn as nn


class EppsPulley(nn.Module):
    """Univariate Epps-Pulley test against a standard normal via numerical
    integration of the empirical characteristic function.

    Input:  (*, N, K)  — N samples across K (sliced) dimensions.
    Output: (*, K)     — per-slice test statistic.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1
        t = torch.linspace(0.0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = (-0.5 * t.square()).exp()
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t  # (*, N, K, n_points)
        cos_mean = torch.cos(x_t).mean(-3)  # (*, K, n_points)
        sin_mean = torch.sin(x_t).mean(-3)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N


class SlicingUnivariateTest(nn.Module):
    """SIGReg: project (N, D) embeddings onto random unit directions and
    apply a univariate Gaussianity test on each slice.

    Input:  (*, N, D)
    Output: scalar (mean over slices) by default.
    """

    def __init__(
        self,
        univariate_test: nn.Module,
        num_slices: int,
        reduction: str = "mean",
        clip_value: float | None = None,
    ):
        super().__init__()
        self.univariate_test = univariate_test
        self.num_slices = num_slices
        self.reduction = reduction
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            seed = int(self.global_step.item())
            g = self._get_generator(x.device, seed)
            A = torch.randn((x.size(-1), self.num_slices), device=x.device, generator=g)
            A = A / A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats = torch.where(stats < self.clip_value, torch.zeros_like(stats), stats)
        if self.reduction == "mean":
            return stats.mean()
        if self.reduction == "sum":
            return stats.sum()
        return stats
