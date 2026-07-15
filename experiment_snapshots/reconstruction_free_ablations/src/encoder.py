import logging

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules import MLP, RMSNormChannels
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
                # "in": EncoderBlock(
                # RMS of input image is reasonable ~0.3 so we Muon
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


class ImpalaBlock(nn.Module):
    def __init__(self, n_of_channels, fan, outscale):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(n_of_channels, n_of_channels, 3, stride=1, padding=1) for _ in range(2)])
        for conv in self.convs:
            trunc_normal_init(conv.weight, fan=fan, scale=outscale)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        for conv in self.convs:
            x = nn.functional.relu(x)
            x = conv(x)
        return x


class ImpalaResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fan, outscale):
        super().__init__()
        self.start_conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        trunc_normal_init(self.start_conv.weight, fan=fan, scale=outscale)
        nn.init.zeros_(self.start_conv.bias)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.impala_blocks = nn.ModuleList([ImpalaBlock(out_channels, fan, outscale) for _ in range(2)])

    def forward(self, x):
        x = self.start_conv(x)
        x = self.pool(x)
        for block in self.impala_blocks:
            x += block(x)
        return x


@gin.configurable
class ImpalaEncoder(nn.Module):
    def __init__(self, C, W, H, fan: str = "in", outscale: float = 1.0, output_dim: int = 4096):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ImpalaResidualBlock(C, 16, fan, outscale),
                ImpalaResidualBlock(16, 32, fan, outscale),
                ImpalaResidualBlock(32, 32, fan, outscale),
            ]
        )
        self.fc = nn.Linear(32 * (W // 8) * (H // 8), 256)
        trunc_normal_init(self.fc.weight, fan=fan, scale=outscale)
        nn.init.zeros_(self.fc.bias)
        self.projection = MLP(256, output_dim, fan=fan, outscale=outscale)
        self.output_dim = output_dim

    def forward(self, x):
        x = _input_transform(x)

        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")

        for block in self.blocks:
            x = block(x)

        x = rearrange(x, "(b t) c h w -> b t (c h w)", b=B)
        x = nn.functional.relu(x)
        x = self.fc(x)
        x = nn.functional.relu(x)
        x = self.projection(x)
        return x


@gin.configurable
class ImpalaEncoderDreamerSize(nn.Module):
    def __init__(self, C, W, H, fan: str = "in", outscale: float = 1.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ImpalaResidualBlock(C, 32, fan, outscale),
                ImpalaResidualBlock(32, 64, fan, outscale),
                ImpalaResidualBlock(64, 128, fan, outscale),
                ImpalaResidualBlock(128, 256, fan, outscale),
            ]
        )
        self.flattend_dim = 256 * (W // 16) * (H // 16)

        self.output_dim = 1024
        self.flatten = nn.Flatten()

        self.final_projector = nn.Sequential(
            nn.Linear(self.flattend_dim, 1024),
            nn.BatchNorm1d(1024, eps=1e-4),
            nn.SiLU(),
            nn.Linear(1024, self.output_dim),
        )

    def forward(self, x):
        x = _input_transform(x)

        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")

        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)
        x = self.final_projector(x)

        return rearrange(x, "(b t) d -> b t d", b=B)


def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    random_tensor.div_(keep_prob)
    return x * random_tensor


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class _ConvNeXtLayerNorm(nn.Module):
    """LayerNorm supporting channels_last (NHWC) and channels_first (NCHW)."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(f"Unknown data_format: {data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class _ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, drop_path_rate: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _ConvNeXtLayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = _GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = _DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


@gin.configurable
class ConvNeXtEncoder(nn.Module):
    """ConvNeXt V2 encoder for 64x64 pixel observations (~3M params with defaults).

    Architecture: Stem(stride-4) -> 4 stages with downsampling -> LN -> Linear.
    """

    def __init__(
        self,
        C: int,
        W: int,
        H: int,
        dims: tuple = (32, 64, 128, 256),
        depths: tuple = (2, 2, 9, 2),
        output_dim: int = 1024,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert len(dims) == len(depths) == 4

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(C, dims[0], kernel_size=4, stride=4),
            _ConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            ds = nn.Sequential(
                _ConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(ds)

        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[_ConvNeXtV2Block(dims[i], drop_path_rate=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        flattend_dim = dims[-1] * 4
        self.head_norm = nn.LayerNorm(flattend_dim, eps=1e-6)
        self.head = nn.Linear(flattend_dim, output_dim)
        self.output_dim = output_dim

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = _input_transform(x)
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.flatten(1)
        x = self.head_norm(x)
        x = self.head(x)
        return rearrange(x, "(b t) d -> b t d", b=B)
