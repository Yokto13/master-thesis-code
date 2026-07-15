from collections import OrderedDict
from typing import Any, Dict

import gin
import torch
import torch.nn as nn
from einops import rearrange, repeat
from modules import MLP, BlockLinear
from torch.distributions import Categorical
from utils import trunc_normal_init, unimix


@gin.configurable
class RSSM(nn.Module):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        action_dim: int,
        embed_dim: int,
        hidden_dim: int,
        unimix_ratio: float = 0.0,
        num_categories: int = 32,
        norm: str = "layernorm",
        fan: str = "in",
        outscale: float = 1.0,
        n_of_blocks: int = 8,
        prior_dropout: float = 0.0,
        posterior_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.unimix_ratio = unimix_ratio

        self.rnn = BlockGRU(
            deter_dim,
            stoch_dim,
            action_dim,
            hidden_dim,
            n_of_dyn_layers=1,
            n_of_blocks=n_of_blocks,
            fan=fan,
            outscale=outscale,
        )

        self.num_categories = num_categories

        assert stoch_dim % num_categories == 0
        # Layer norm is essential for categorical RSSM
        # Without it, the logits go haywire and softmax saturates
        self.prior_net = nn.Sequential(
            OrderedDict(
                [
                    (
                        "features",
                        MLP(
                            deter_dim,
                            hidden_dim,
                            num_layers=2,
                            fan=fan,
                            outscale=outscale,
                            norm=norm,
                            dropout=prior_dropout,
                        ),
                    ),
                    ("out", nn.Linear(hidden_dim, stoch_dim)),
                ]
            )
        )

        self.posterior_net = nn.Sequential(
            OrderedDict(
                [
                    (
                        "features",
                        MLP(
                            deter_dim + embed_dim,
                            hidden_dim,
                            num_layers=1,
                            fan=fan,
                            outscale=outscale,
                            norm=norm,
                            dropout=posterior_dropout,
                        ),
                    ),
                    ("out", nn.Linear(hidden_dim, stoch_dim)),
                ]
            )
        )

        # Apply trunc_normal_init to all layers
        self._init_weights(fan, outscale)

    def _init_weights(self, fan: str, outscale: float):
        # Initialize final Linear layers in prior_net and posterior_net
        # (MLP layers are already initialized in MLP.__init__)
        for net in [self.prior_net, self.posterior_net]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    trunc_normal_init(layer.weight, fan=fan, scale=outscale)
                    nn.init.zeros_(layer.bias)

    def deter_step(self, stochastic_state, action, prev_deter_state):
        return self.rnn(prev_deter_state, stochastic_state, action)

    def stoch_step(self, deter_state, latent=None) -> Dict[str, torch.Tensor]:
        if latent is None:
            logits = self.prior_net(deter_state)
        else:
            x = torch.cat([deter_state, latent], dim=-1)
            logits = self.posterior_net(x)

        logits = rearrange(logits, "b (c k) -> b c k", c=self.num_categories)

        probs = unimix(logits, self.unimix_ratio)
        num_classes = self.stoch_dim // self.num_categories
        sample = nn.functional.one_hot(Categorical(probs=probs).sample(), num_classes=num_classes)
        sample = sample + probs - probs.detach()

        raw_logits = rearrange(logits, "b c k -> b (c k)")
        sample = rearrange(sample, "b c k -> b (c k)")

        return {"logits": raw_logits, "sample": sample}

    @torch.compile(fullgraph=True)
    def wake_step(self, stochastic_state, action, prev_deter_state, embedded_image) -> Dict[str, Any]:
        deter_state = self.deter_step(stochastic_state, action, prev_deter_state)

        prior = self.stoch_step(deter_state)
        posterior = self.stoch_step(deter_state, latent=embedded_image)

        return {"prior": prior, "posterior": posterior, "deter_state": deter_state}

    def dream_step(self, stochastic_state, action, prev_deter_state) -> Dict[str, Any]:
        deter_state = self.deter_step(stochastic_state, action, prev_deter_state)

        prior = self.stoch_step(deter_state)
        return {"prior": prior, "deter_state": deter_state}

    def init_state(self, batch_size, device) -> Dict[str, torch.Tensor]:
        """
        Helper to create the initial zero state.
        """
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def forward(self, actions, embeds, is_first, state=None) -> Dict[str, Any]:
        posterior = {"sample": [], "logits": []}
        prior = {"sample": [], "logits": []}
        deter_states = []

        T = actions.shape[1]
        B = actions.shape[0]

        if state is None:
            state = self.init_state(B, actions.device)

        prev_deter = state["deter"]
        prev_stoch = state["stoch"]

        for t in range(T):
            mask = (1.0 - is_first[:, t].float()).unsqueeze(-1)
            prev_deter = prev_deter * mask
            prev_stoch = prev_stoch * mask

            step = self.wake_step(prev_stoch, actions[:, t], prev_deter, embeds[:, t])

            prior["sample"].append(step["prior"]["sample"])
            prior["logits"].append(step["prior"]["logits"])
            posterior["sample"].append(step["posterior"]["sample"])
            posterior["logits"].append(step["posterior"]["logits"])

            deter = step["deter_state"]
            deter_states.append(deter)

            prev_deter = deter
            prev_stoch = step["posterior"]["sample"]

        stacked_posterior = {key: torch.stack(values, dim=1) for key, values in posterior.items()}
        stacked_prior = {key: torch.stack(values, dim=1) for key, values in prior.items()}

        return {
            "posterior": stacked_posterior,
            "prior": stacked_prior,
            "deter_state": torch.stack(deter_states, dim=1),
            "last_state": {"deter": prev_deter, "stoch": prev_stoch},
        }


class BlockGRU(nn.Module):
    def __init__(
        self,
        deter_dim,
        stoch_dim,
        action_dim,
        hidden_dim,
        n_of_dyn_layers,
        n_of_blocks,
        fan: str = "in",
        outscale: float = 1.0,
        norm: str = "rmsnorm",
        update_bias: int = -1,
    ):
        """A GRU implementation based on the one in Hafner's RSSM, more complex than standard GRU."""
        super().__init__()
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.update_bias = update_bias
        self.g = n_of_blocks
        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        self.n_of_dyn_layers = n_of_dyn_layers

        assert norm in ("none", "layernorm", "rmsnorm")

        norm_fn = {
            "none": lambda dim: nn.Identity(),
            "layernorm": lambda dim: nn.LayerNorm(dim, eps=1e-4),
            "rmsnorm": lambda dim: nn.RMSNorm(dim, eps=1e-4),
        }[norm]

        self.deter_transform = MLP(deter_dim, hidden_dim)
        self.stoch_transform = MLP(stoch_dim, hidden_dim)
        self.action_transform = MLP(action_dim, hidden_dim)

        first_layer_in_features = deter_dim + 3 * self.g * hidden_dim
        current_in_features = first_layer_in_features

        layers = []
        for _ in range(n_of_dyn_layers):
            layers.append(BlockLinear(current_in_features, deter_dim, blocks=n_of_blocks, fan=fan, outscale=outscale))
            layers.append(norm_fn(deter_dim))
            layers.append(nn.SiLU())
            current_in_features = deter_dim

        self.dyn_layers = nn.Sequential(*layers)

        self.dyn_to_gru = BlockLinear(deter_dim, 3 * deter_dim, blocks=n_of_blocks, fan=fan, outscale=outscale)

    @torch.compile(fullgraph=True)
    def forward(self, deter, stoch, action):
        x0 = self.deter_transform(deter)
        x1 = self.stoch_transform(stoch)
        x2 = self.action_transform(action)
        x = torch.cat([x0, x1, x2], dim=-1)
        x = repeat(x, "... d -> ... g d", g=self.g)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], dim=-1))

        x = self.dyn_layers(x)
        gru_out = self.flat2group(self.dyn_to_gru(x))
        reset, cand, update = (self.group2flat(g) for g in torch.chunk(gru_out, 3, dim=-1))
        reset = torch.sigmoid(reset)
        cand = torch.tanh(cand * reset)
        update = torch.sigmoid(update + self.update_bias)
        deter = update * cand + (1 - update) * deter
        return deter

    def flat2group(self, x):
        return rearrange(x, "... (g h) -> ... g h", g=self.g)

    def group2flat(self, x):
        return rearrange(x, "... g h -> ... (g h)", g=self.g)
