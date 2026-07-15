import logging

import gin
import torch.nn as nn
from einops import rearrange
from modules import RMSNormChannels
from utils import trunc_normal_init

logger = logging.getLogger(__name__)


def _input_transform(x):
    return x.float() / 255.0 - 0.5


class Conv2dMaxPoolBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        fan: str = "in",
        outscale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        trunc_normal_init(self.conv.weight, fan=fan, scale=outscale)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=1,
        fan: str = "in",
        outscale: float = 1.0,
        norm="rms",
        activation="silu",
    ):
        super().__init__()
        bias = norm == "rms"
        if stride > 1:
            logger.info("Using stride > 1 in EncoderBlock.")
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            # Apply trunc_normal_init to Conv2d layer
            trunc_normal_init(self.conv.weight, fan=fan, scale=outscale)
            if bias:
                nn.init.zeros_(self.conv.bias)
        else:
            # If stride=1 we use maxpooling to downsample.
            self.conv = Conv2dMaxPoolBlock(
                in_channels, out_channels, kernel_size, padding, fan=fan, outscale=outscale, bias=bias
            )

        match norm:
            case "group":
                self.norm = nn.GroupNorm(1, out_channels)
            case "rms":
                self.norm = RMSNormChannels(out_channels)
            case "batch":
                self.norm = nn.BatchNorm2d(out_channels, eps=1e-4)
            case _:
                raise ValueError(f"Unknown norm: {norm}")
        # but in config.yaml overwrites to SiLU
        match activation:
            case "silu":
                self.activation = nn.SiLU()
            case "gelu":
                self.activation = nn.GELU()
            case _:
                raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


@gin.configurable
class Encoder(nn.Module):
    def __init__(self, C, W, H, fan: str = "in", outscale: float = 1.0, norm: str = "rms", stride=1, depth=64):
        super().__init__()

        # Original JAX depths based on depth=64 * mults(2,3,4,4)
        K = 5
        mults = [2, 3, 4, 4]
        depths = [depth * m for m in mults]
        # TODO: We should check how Dreamer pads...
        padding = (K - 1) // 2  # To maintain spatial dimensions before pooling

        import torch.nn as nn

        self.blocks = nn.ModuleDict(
            {
                "stage_1": EncoderBlock(
                    C, depths[0], kernel_size=K, fan=fan, stride=stride, padding=padding, outscale=outscale, norm=norm
                ),
                "stage_2": EncoderBlock(
                    depths[0],
                    depths[1],
                    kernel_size=K,
                    fan=fan,
                    stride=stride,
                    padding=padding,
                    outscale=outscale,
                    norm=norm,
                ),
                "stage_3": EncoderBlock(
                    depths[1],
                    depths[2],
                    kernel_size=K,
                    fan=fan,
                    stride=stride,
                    padding=padding,
                    outscale=outscale,
                    norm=norm,
                ),
                "stage_4": EncoderBlock(
                    depths[2],
                    depths[3],
                    kernel_size=K,
                    fan=fan,
                    stride=stride,
                    padding=padding,
                    outscale=outscale,
                    norm=norm,
                ),
            }
        )

        # self.blocks = self.blocks.to(memory_format=torch.channels_last)

        self.flatten = nn.Flatten()

        # Expose output size so downstream modules (RSSM) can use it
        final_w = W // 16
        final_h = H // 16
        self.output_dim = 1024

        self.final_projector = nn.Sequential(
            nn.Linear(depths[-1] * final_w * final_h, 1024),
            nn.BatchNorm1d(1024, eps=1e-4),
            nn.SiLU(),
            nn.Linear(1024, self.output_dim),
        )

    def forward(self, x):
        x = _input_transform(x)

        # Merge Batch and Time
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")

        # x = x.to(memory_format=torch.channels_last)

        for block in self.blocks.values():
            x = block(x)

        x = self.flatten(x)

        x = self.final_projector(x)

        # Restore (B, T, D)
        return rearrange(x, "(b t) d -> b t d", b=B)
