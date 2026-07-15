from typing import Callable, Dict

import torch
from torch.optim import Optimizer


class CompositeOptimizer(Optimizer):
    def __init__(
        self,
        constructors: Dict[str, Callable],
        params: Dict[str, Dict],
    ):
        assert set(constructors.keys()) == set(params.keys()), "Constructors and params keys must match"

        self._keys = list(params.keys())
        self.param_groups = [params[key] for key in self._keys]
        super().__init__(self.param_groups, defaults={})

        self._optimizers = {}
        for key in self._keys:
            self._optimizers[key] = constructors[key](**params[key])

        for opt in self._optimizers.values():
            opt.state = self.state

    def load_state_dict(self, state_dict):
        # PyTorch's load_state_dict reassigns self.state to a new dict,
        # which breaks the shared reference. Re-share after loading.
        super().load_state_dict(state_dict)
        for opt in self._optimizers.values():
            opt.state = self.state

    @staticmethod
    def _sync_group_hparams(parent_group: Dict, child_opt: Optimizer):
        # Keep child optimizer hyperparameters schedulable from the parent optimizer.
        for child_group in child_opt.param_groups:
            for key, value in parent_group.items():
                if key in ("params", "kind"):
                    continue
                if key in child_group:
                    child_group[key] = value

    @torch.no_grad()
    def step(self):
        for idx, key in enumerate(self._keys):
            opt = self._optimizers[key]
            parent_group = self.param_groups[idx]
            self._sync_group_hparams(parent_group, opt)
            opt.step()
