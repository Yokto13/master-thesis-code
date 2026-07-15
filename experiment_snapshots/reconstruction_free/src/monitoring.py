"""
Diagnostic monitoring for the Muon-vs-LaProp comparison.

Pieces:
- WandbVisualizer: monitorch AbstractVisualizer that routes lens output to wandb
- UpdateNormTracker: per-parameter update norms via optimizer pre/post step hooks,
  classified into the 'muon' vs 'laprop' group using the same exclusion keywords as
  utils.get_optimizer_param_groups
- BlockGRUBlockTracker: per-block update norms for BlockLinear.kernel weights
- MonitoringSuite: bundles everything; trainer calls .step_logged(global_step)
  at log_interval boundaries to push metrics to wandb
"""

from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from optim.composite_optimizer import CompositeOptimizer

import wandb

try:
    from monitorch.inspector import PyTorchInspector
    from monitorch.lens import ParameterGradientGeometry, ParameterNorm
    from monitorch.visualizer import AbstractVisualizer

    _MONITORCH_AVAILABLE = True
except Exception:
    _MONITORCH_AVAILABLE = False
    AbstractVisualizer = object  # type: ignore


_EXCLUDE_KEYWORDS = ("in", "out", "head", "dyn_to_gru")

# Top-level dreamer submodules to expose to monitorch. We deliberately omit
# `target_critic` because its parameters are frozen (requires_grad=False) and
# monitorch's gradient lens tries to register a post-accumulate-grad hook,
# which fails on frozen tensors.
_MONITORED_SUBMODULES = ("encoder", "decoder", "rssm", "rewards", "discounts", "actor", "critic")


class _FilteredView(nn.Module):
    """Container that re-exposes a subset of another module's children, by reference."""

    def __init__(self, source: nn.Module, child_names):
        super().__init__()
        for name in child_names:
            if hasattr(source, name):
                setattr(self, name, getattr(source, name))


def _has_only_grad_params(module: nn.Module) -> bool:
    """True if every parameter (recursively) under this module has requires_grad=True."""
    return all(p.requires_grad for p in module.parameters())


def _classify(param_name: str) -> str:
    """Same exclusion logic as utils.get_optimizer_param_groups, returning the optimizer group label."""
    if any(kw in param_name for kw in _EXCLUDE_KEYWORDS):
        return "laprop"
    return "muon"


class WandbVisualizer(AbstractVisualizer):  # type: ignore[misc]
    """
    monitorch visualizer that buffers values and flushes to wandb on demand.

    monitorch ticks once per "epoch" (we call tick() at log_interval boundaries).
    On each tick the lenses call plot_numerical_values etc. with the latest
    aggregates. We translate (main_tag, module_name, agg) into a wandb key and
    buffer it; the suite flushes the buffer.
    """

    def __init__(self):
        self._buffer: dict[str, float] = {}

    def register_tags(self, main_tag: str, tag_attr) -> None:  # noqa: D401
        pass

    @staticmethod
    def _key(prefix: str, main_tag: str, module_name: str, subtag: str) -> str:
        slug = f"{main_tag}/{module_name}/{subtag}".lower().replace(" ", "_")
        return f"{prefix}/{slug}"

    def plot_numerical_values(self, epoch, main_tag, values_dict, ranges_dict=None) -> None:
        for module_name, agg_dict in values_dict.items():
            group = _classify(module_name)
            for agg_name, value in agg_dict.items():
                k = self._key(f"monitorch/{group}", main_tag, module_name, agg_name)
                self._buffer[k] = float(value)
        if ranges_dict:
            for module_name, agg_dict in ranges_dict.items():
                group = _classify(module_name)
                for (a, b), (va, vb) in agg_dict.items():
                    self._buffer[self._key(f"monitorch/{group}", main_tag, module_name, a)] = float(va)
                    self._buffer[self._key(f"monitorch/{group}", main_tag, module_name, b)] = float(vb)

    def plot_probabilities(self, epoch, main_tag, values_dict) -> None:
        for module_name, agg_dict in values_dict.items():
            group = _classify(module_name)
            for agg_name, value in agg_dict.items():
                self._buffer[self._key(f"monitorch/{group}", main_tag, module_name, agg_name)] = float(value)

    def plot_relations(self, epoch, main_tag, values_dict) -> None:
        for tag, relations in values_dict.items():
            for sub, value in relations.items():
                group = _classify(sub)
                self._buffer[self._key(f"monitorch/{group}", main_tag, tag, sub)] = float(value)

    def drain(self) -> dict[str, float]:
        out = self._buffer
        self._buffer = {}
        return out


class UpdateNormTracker:
    """
    Per-parameter update norms (||Δw||, ||w||, ratio) classified into muon/laprop groups.

    Registers pre/post step hooks on the relevant optimizer(s). For CompositeOptimizer
    we hook each inner optimizer separately so the snapshot diff cleanly captures only
    that optimizer's update.

    Accumulates values across multiple training steps; flush() returns the windowed mean
    and resets.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self._id_to_name = {id(p): n for n, p in model.named_parameters()}
        self._snapshot: dict[int, torch.Tensor] = {}
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

        if isinstance(optimizer, CompositeOptimizer):
            for label, inner in optimizer._optimizers.items():
                inner.register_step_pre_hook(self._pre_hook)
                inner.register_step_post_hook(lambda opt, args, kw, lbl=label: self._post_hook(opt, lbl))
        else:
            optimizer.register_step_pre_hook(self._pre_hook)
            optimizer.register_step_post_hook(lambda opt, args, kw: self._post_hook(opt, "laprop"))

    def _pre_hook(self, opt, *args, **kwargs):
        for group in opt.param_groups:
            for p in group["params"]:
                if id(p) in self._id_to_name:
                    self._snapshot[id(p)] = p.detach().clone()

    @torch.no_grad()
    def _post_hook(self, opt, optim_label: str):
        for group in opt.param_groups:
            for p in group["params"]:
                pid = id(p)
                name = self._id_to_name.get(pid)
                if name is None:
                    continue
                prev = self._snapshot.pop(pid, None)
                if prev is None:
                    continue
                delta = (p.detach() - prev).float()
                w = p.detach().float()
                update_norm = delta.norm().item()
                weight_norm = w.norm().item()
                ratio = update_norm / max(weight_norm, 1e-12)

                # The *param classification* uses our exclusion keywords applied to the
                # full named-parameters path. It should match optim_label except for ndim<2
                # params that always go to laprop regardless of name.
                group_label = _classify(name)
                base = f"update/{group_label}/{name}"
                self._sums[f"{base}/update_norm"] += update_norm
                self._sums[f"{base}/weight_norm"] += weight_norm
                self._sums[f"{base}/update_to_weight"] += ratio
                self._counts[f"{base}/update_norm"] += 1
                self._counts[f"{base}/weight_norm"] += 1
                self._counts[f"{base}/update_to_weight"] += 1

    def flush(self) -> dict[str, float]:
        out = {}
        for k, total in self._sums.items():
            n = self._counts[k]
            if n > 0:
                out[k] = total / n
        self._sums.clear()
        self._counts.clear()
        return out


class BlockGRUBlockTracker:
    """
    Per-block update + weight norms for BlockLinear.kernel tensors.

    BlockLinear.kernel has shape (G, in/G, out/G) — first dim is the block index.
    Reports ||Δw_g||, ||w_g||, and their ratio for each g, plus the across-block
    spread (max/min) so we can see if some blocks are getting hammered harder.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        # Lazy import to avoid circular dep
        from modules import BlockLinear

        # Find every BlockLinear module within the model and remember its qualified name
        self._modules: list[tuple[str, BlockLinear]] = []
        for name, module in model.named_modules():
            if isinstance(module, BlockLinear):
                self._modules.append((name, module))

        self._param_id_to_owner = {id(m.kernel): name for name, m in self._modules}
        self._snapshot: dict[int, torch.Tensor] = {}
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

        if isinstance(optimizer, CompositeOptimizer):
            for inner in optimizer._optimizers.values():
                inner.register_step_pre_hook(self._pre_hook)
                inner.register_step_post_hook(lambda opt, args, kw: self._post_hook(opt))
        else:
            optimizer.register_step_pre_hook(self._pre_hook)
            optimizer.register_step_post_hook(lambda opt, args, kw: self._post_hook(opt))

    def _pre_hook(self, opt, *args, **kwargs):
        for group in opt.param_groups:
            for p in group["params"]:
                if id(p) in self._param_id_to_owner:
                    self._snapshot[id(p)] = p.detach().clone()

    @torch.no_grad()
    def _post_hook(self, opt):
        for group in opt.param_groups:
            for p in group["params"]:
                pid = id(p)
                owner = self._param_id_to_owner.get(pid)
                if owner is None:
                    continue
                prev = self._snapshot.pop(pid, None)
                if prev is None:
                    continue
                delta = (p.detach() - prev).float()
                w = p.detach().float()
                # Per-block (axis 0) norms
                un_per_block = delta.flatten(1).norm(dim=1)
                wn_per_block = w.flatten(1).norm(dim=1)
                ratio_per_block = un_per_block / wn_per_block.clamp(min=1e-12)

                base = f"blockgru/{owner}"
                un_max = un_per_block.max().item()
                un_min = un_per_block.min().item()
                un_mean = un_per_block.mean().item()
                un_std = un_per_block.std().item() if un_per_block.numel() > 1 else 0.0
                ratio_max = ratio_per_block.max().item()
                ratio_mean = ratio_per_block.mean().item()

                for k, v in [
                    ("update_norm/max", un_max),
                    ("update_norm/min", un_min),
                    ("update_norm/mean", un_mean),
                    ("update_norm/std", un_std),
                    ("update_to_weight/max", ratio_max),
                    ("update_to_weight/mean", ratio_mean),
                ]:
                    full = f"{base}/{k}"
                    self._sums[full] += v
                    self._counts[full] += 1

                # Per-block detail (one scalar per block per metric)
                for g_idx in range(un_per_block.numel()):
                    detail = f"{base}/per_block/block_{g_idx}"
                    self._sums[f"{detail}/update_norm"] += un_per_block[g_idx].item()
                    self._counts[f"{detail}/update_norm"] += 1
                    self._sums[f"{detail}/update_to_weight"] += ratio_per_block[g_idx].item()
                    self._counts[f"{detail}/update_to_weight"] += 1

    def flush(self) -> dict[str, float]:
        out = {}
        for k, total in self._sums.items():
            n = self._counts[k]
            if n > 0:
                out[k] = total / n
        self._sums.clear()
        self._counts.clear()
        return out


class MonitoringSuite:
    """
    Trainer-facing entry point. Owns the monitorch inspector and our two custom trackers.

    Usage:
        suite = MonitoringSuite(dreamer, dreamer.optimizer)
        ...
        # In the training loop, when ready to log:
        suite.log_to_wandb(global_step)
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.update_tracker = UpdateNormTracker(model, optimizer)
        self.block_tracker = BlockGRUBlockTracker(model, optimizer)

        self._inspector: Optional["PyTorchInspector"] = None
        self._wandb_vis: Optional[WandbVisualizer] = None
        if _MONITORCH_AVAILABLE:
            self._wandb_vis = WandbVisualizer()
            # Skip frozen submodules — monitorch's gradient lens registers
            # post-accumulate-grad hooks that error on frozen parameters
            # (notably target_critic which has requires_grad=False).
            kept = [n for n in _MONITORED_SUBMODULES if hasattr(model, n) and _has_only_grad_params(getattr(model, n))]
            view = _FilteredView(model, kept)
            self._inspector = PyTorchInspector(
                lenses=[
                    ParameterNorm(parameters=("weight",), comparison_plot=False),
                    ParameterNorm(parameters=("kernel",), comparison_plot=False),
                    ParameterGradientGeometry(parameters=("weight",), compute_correlation=False),
                    ParameterGradientGeometry(parameters=("kernel",), compute_correlation=False),
                ],
                visualizer=self._wandb_vis,
                module=view,
            )

    def log_to_wandb(self, global_step: int) -> None:
        log: dict[str, float] = {}
        log.update(self.update_tracker.flush())
        log.update(self.block_tracker.flush())
        if self._inspector is not None and self._wandb_vis is not None:
            # tick_epoch invokes the lenses to collect+aggregate+visualize. Our
            # WandbVisualizer just buffers; we drain after.
            self._inspector.tick_epoch(epoch=global_step)
            log.update(self._wandb_vis.drain())
        if log:
            wandb.log(log, step=global_step)
