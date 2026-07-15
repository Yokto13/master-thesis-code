import logging

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules import BlockLinear, RMSNormChannels
from utils import trunc_normal_init

logger = logging.getLogger(__name__)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        fan: str = "in",
        outscale: float = 1.0,
        activation="silu",
        norm: str = "rms",
        dropout: float = 0.0,
    ):
        super().__init__()
        # JAX equivalent: Upsample -> Conv5x5 -> Norm -> Act
        # Bias is only used with RMS norm (no affine params); batch/group norm have their own bias.
        bias = norm == "rms" and activation is not None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=bias)
        # DreamerV3 source code defaults to GELU for encoder
        # but in config.yaml overwrites to SiLU
        self.activation_type = activation
        if activation is not None:
            activation_fn = {"silu": nn.SiLU, "gelu": nn.GELU}[activation]
            self.act = activation_fn()
            match norm:
                case "group":
                    self.norm = nn.GroupNorm(1, out_channels)
                case "rms":
                    self.norm = RMSNormChannels(out_channels)
                case "batch":
                    self.norm = nn.BatchNorm2d(out_channels, eps=1e-4)
                case _:
                    raise ValueError(f"Unknown norm: {norm}")
            if bias:
                nn.init.zeros_(self.conv.bias)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

        # Apply trunc_normal_init to Conv2d layer
        trunc_normal_init(self.conv.weight, fan=fan, scale=outscale)

    def forward(self, x):
        # 1. Nearest Neighbor Upsampling (2x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # 2. Convolution
        x = self.conv(x)
        # 3. Norm + Act
        if self.activation_type is not None:
            x = self.norm(x)
            x = self.act(x)
            x = self.dropout(x) if self.dropout is not None else x
        return x


@gin.configurable
class Decoder(nn.Module):
    def __init__(
        self,
        C,
        W,
        H,
        stoch_dim: int,
        deter_dim: int | None,
        hidden_dim: int,
        fan: str = "in",
        outscale: float = 1.0,
        norm: str = "rms",
        blocks: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        depth = 64
        mults = [2, 3, 4, 4]
        depths = [depth * m for m in mults]
        factor = 2 ** len(depths)
        self.minres = (W // factor, H // factor)
        assert 3 <= self.minres[0] <= 16, self.minres
        assert 3 <= self.minres[1] <= 16, self.minres

        flattend_size = depths[-1] * self.minres[0] * self.minres[1]
        self.bspace = blocks
        # deter_dim=None builds a stoch-only decoder (used as a probe on raw encoder embeddings).
        self.deter_projection = (
            BlockLinear(deter_dim, flattend_size, blocks=blocks, fan=fan, outscale=outscale)
            if deter_dim is not None
            else None
        )
        self.stoch_projection = nn.Sequential(
            # Give it more capacity with expansion pattern as stoch is typically small
            nn.Linear(stoch_dim, 2 * hidden_dim),
            nn.RMSNorm(2 * hidden_dim, eps=1e-04, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, flattend_size),
        )

        match norm:
            case "group":
                self.spatial_norm = nn.GroupNorm(1, depths[-1])
            case "rms":
                self.spatial_norm = RMSNormChannels(depths[-1])
            case "batch":
                self.spatial_norm = nn.BatchNorm2d(depths[-1], eps=1e-4)
            case _:
                raise ValueError(f"Unknown norm: {norm}")
        self.spatial_act = nn.SiLU()

        # 2. Main Upsampling Stack
        self.blocks = nn.ModuleDict(
            {
                "stage_1": DecoderBlock(
                    depths[-1], depths[-2], fan=fan, outscale=outscale, norm=norm, dropout=dropout
                ),
                "stage_2": DecoderBlock(
                    depths[-2], depths[-3], fan=fan, outscale=outscale, norm=norm, dropout=dropout
                ),
                "stage_3": DecoderBlock(
                    depths[-3], depths[-4], fan=fan, outscale=outscale, norm=norm, dropout=dropout
                ),
                "out": DecoderBlock(depths[-4], C, fan=fan, outscale=outscale, activation=None, norm=norm),
            }
        )
        # self.blocks = self.blocks.to(memory_format=torch.channels_last)
        self._init_weights(fan, outscale)

    def _init_weights(self, fan, outscale):
        # init stoch projection
        for m in self.stoch_projection.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, fan=fan, scale=outscale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, stoch: torch.Tensor, deter: torch.Tensor | None = None) -> torch.Tensor:
        B, T = stoch.shape[0], stoch.shape[1]
        stoch = rearrange(stoch, "B T D -> (B T) D")

        x1 = self.stoch_projection(stoch)
        x = rearrange(x1, "BT (C H W) -> BT C H W", H=self.minres[0], W=self.minres[1])
        if deter is not None:
            assert self.deter_projection is not None, "Decoder was built without a deter pathway"
            deter = rearrange(deter, "B T D -> (B T) D")
            x0 = self.deter_projection(deter)
            x0 = rearrange(x0, "BT (g H W c) -> BT (g c) H W", g=self.bspace, H=self.minres[0], W=self.minres[1])
            x = x + x0
        x = self.spatial_norm(x)
        x = self.spatial_act(x)
        # x = x.to(memory_format=torch.channels_last)

        for b in self.blocks.values():
            x = b(x)

        x = torch.sigmoid(x)
        x = rearrange(x, "(B T) C H W -> B T C H W", B=B, T=T)
        return x

    def reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target / 255.0
        loss = F.mse_loss(recon, target, reduction="none")
        loss = loss.sum(dim=[2, 3, 4])
        return loss.mean()
