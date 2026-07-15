import torch
from torch.optim import Muon, Optimizer

from .laprop import LaProp


class LaPropMuon(Optimizer):
    def __init__(
        self,
        laprop_params,
        muon_params,
        laprop_lr=4e-4,
        laprop_betas=(0.9, 0.999),
        laprop_eps=1e-15,
        muon_lr=4e-4,
        muon_weight_decay=0.0,
        muon_momentum=0.95,
        muon_adjust_lr_fn=None,
    ):
        laprop_list = list(laprop_params)
        muon_list = list(muon_params)

        param_groups = [
            {
                "kind": "laprop",
                "params": laprop_list,
                "lr": laprop_lr,
                "betas": laprop_betas,
                "eps": laprop_eps,
            },
            {
                "kind": "muon",
                "params": muon_list,
                "lr": muon_lr,
                "weight_decay": muon_weight_decay,
                "momentum": muon_momentum,
            },
        ]
        super().__init__(param_groups, defaults={})

        self._laprop = LaProp(laprop_list, lr=laprop_lr, betas=laprop_betas, eps=laprop_eps)
        self._muon = Muon(
            muon_list,
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            adjust_lr_fn=muon_adjust_lr_fn,
        )

        # Share state so all optimizer state lives in self.state (one source of truth).
        # This makes state_dict() / load_state_dict() work on the parent.
        self._laprop.state = self.state
        self._muon.state = self.state

    @torch.no_grad()
    def step(self):
        # Sync schedulable hyperparams from our param_groups to the sub-optimizers
        self._laprop.param_groups[0]["lr"] = self.param_groups[0]["lr"]
        self._muon.param_groups[0]["lr"] = self.param_groups[1]["lr"]

        self._laprop.step()
        self._muon.step()

    def load_state_dict(self, state_dict):
        # PyTorch's load_state_dict reassigns self.state to a new dict,
        # which breaks the shared reference. Re-share after loading.
        super().load_state_dict(state_dict)
        self._laprop.state = self.state
        self._muon.state = self.state
