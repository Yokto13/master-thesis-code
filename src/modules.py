import gin
import torch
import torch.nn as nn
from utils import symexp, trunc_normal_init, unimix


class BlockLinear(nn.Module):
    def __init__(self, in_features, out_features, blocks, fan="in", outscale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.bias = nn.Parameter(torch.zeros(out_features))

        kernel_shape = (blocks, in_features // blocks, out_features // blocks)
        self.kernel = nn.Parameter(torch.empty(kernel_shape))

        self._init_weights(fan=fan, outscale=outscale)

    def _init_weights(self, fan: str, outscale: float):
        # Kernel shape is (G, I/G, O/G). The original DreamerV3 JAX computes fan
        # as space=prod(shape[:-2])=G, giving fan_in=I, fan_out=O. Our trunc_normal_init
        # uses PyTorch conv conventions (space=prod(shape[2:])) which gives wrong fan
        # for this layout, so we pass the correct (I, O) shape explicitly.
        trunc_normal_init(self.kernel, fan=fan, scale=outscale, fan_shape=(self.in_features, self.out_features))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        assert x.shape[-1] % self.blocks == 0
        x = x.reshape((*x.shape[:-1], self.blocks, self.in_features // self.blocks))
        x = torch.einsum("...ki,kio->...ko", x, self.kernel)
        x = x.reshape((*x.shape[:-2], self.out_features))
        x = x + self.bias
        return x


@gin.configurable
class MLP(nn.Module):
    """
    Unified MLP backbone for feed-forward networks.

    Structure: (Linear → Norm (optional) → Activation) × num_layers

    Output dimension equals hidden_dim. Use a separate Linear head
    to project to the final required dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        activation: str = "silu",
        norm: str = "layernorm",
        fan: str = "in",
        outscale: float = 1.0,
        blocks: int = 1,
    ):
        super().__init__()

        assert num_layers >= 1, "MLP requires at least 1 layer"
        assert norm in ("none", "layernorm", "rmsnorm")

        activation_fn = {"silu": nn.SiLU, "gelu": nn.GELU}[activation]
        norm_fn = {
            "none": lambda dim: nn.Identity(),
            "layernorm": lambda dim: nn.LayerNorm(dim, eps=1e-4),
            "rmsnorm": lambda dim: nn.RMSNorm(dim, eps=1e-4),
        }[norm]

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            if blocks > 1:
                layers.append(BlockLinear(in_dim, hidden_dim, blocks, fan=fan, outscale=outscale))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(norm_fn(hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dim

        # Apply trunc_normal_init to all Linear layers
        self._init_weights(fan, outscale)

    def _init_weights(self, fan: str, outscale: float):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                trunc_normal_init(layer.weight, fan=fan, scale=outscale)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class DenseHead(nn.Module):
    """
    Used for Reward and Discount heads.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        output_dim=1,
        num_hidden_layers=3,
        fan: str = "in",
        outscale: float = 1.0,
        norm: str = "layernorm",
    ):
        super().__init__()
        self.backbone = MLP(input_dim, hidden_dim, num_layers=num_hidden_layers, fan=fan, outscale=1.0, norm=norm)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.mse = torch.nn.MSELoss(reduction="none")

        # Apply trunc_normal_init to final head layer with specified outscale
        trunc_normal_init(self.head.weight, fan=fan, scale=outscale)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.backbone(x))

    def mse_loss(self, predictions, targets, reduction="mean"):
        mse = self.mse(predictions, targets)
        if reduction == "mean":
            mse = mse.mean()
        return mse

    def loss(self, predictions, targets, reduction="mean"):
        return self.mse_loss(predictions, targets, reduction=reduction)


class TwoHotHead(DenseHead):
    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        num_hidden_layers=3,
        buckets_n=255,
        fan: str = "in",
        outscale: float = 1.0,
        norm: str = "layernorm",
    ):
        # Initialize parent DenseHead with bins as output dimension
        super().__init__(input_dim, hidden_dim, buckets_n, num_hidden_layers, fan=fan, outscale=outscale, norm=norm)

        # Initialize and register bucket values for two-hot encoding
        buckets = self.init_buckets(buckets_n)
        self.register_buffer("buckets", buckets)

        self._ce = torch.nn.CrossEntropyLoss(reduction="none")

    def init_buckets(self, buckets_n):
        """
        Initialize bucket positions for two-hot encoding using symexp transformation.

        An implementation trick from the DreamerV3 codebase.

        Creates symmetric buckets around **zero**:
        - For odd number of buckets: includes exact zero bucket
            - For odd numbers (255) it is probably unecessary
        - For even number of buckets: symmetric around zero without zero bucket

        Args:
            buckets_n: Number of buckets to create

        Returns:
            Tensor of bucket positions
        """
        if buckets_n % 2 == 1:
            half = torch.linspace(-20, 0, (buckets_n - 1) // 2 + 1, dtype=torch.float32)
            half = symexp(half)
            buckets = torch.cat([half, -half[:-1].flip(0)], dim=0)
        else:
            half = torch.linspace(-20, 0, buckets_n // 2, dtype=torch.float32)
            half = symexp(half)
            buckets = torch.cat([half, -half.flip(0)], dim=0)

        return buckets

    def forward(self, x, return_logits=False):
        logits = self.head(self.backbone(x))
        probs = torch.softmax(logits, dim=-1)
        # Predictions are read out as the weighted average of the bin positions weighted by their predicted
        # probabilities.

        # DreamerV3 code notes that naive probs @ self.buckets is unstable.
        # The problem is that naively probs @ buckets sums numbers with vastly different magnitudes which
        # especially in lower precision results in numerical errors. The code in dreamer_two_hot_splitting.py
        # shows that the naive way can result in errors of up to 1 in float32.
        # This is probably not a big deal for value prediction which on our problems is typically dozens or hundreds
        # but could screw rewards head. Additionally, let not take any chances.
        # The way to fix that is use the symmetric nature of the buckets and notice that buckets[k] = -buckets[-k-1].
        # Thus we can rearrange the sum to group terms of similar magnitude together.

        n = len(self.buckets)
        m = (n - 1) // 2
        if n % 2 == 1:
            # Odd number of buckets, includes exact zero bucket
            p1 = probs[..., :m]
            p2 = probs[..., m : m + 1]
            p3 = probs[..., m + 1 :]

            b1 = self.buckets[:m]
            b2 = self.buckets[m : m + 1]
            b3 = self.buckets[m + 1 :]
            v = (p2 * b2).sum(-1) + (torch.flip(p1 * b1, dims=[-1]) + (p3 * b3)).sum(-1)
        else:
            # Even number of buckets, symmetric around zero without zero bucket
            p1 = probs[..., :m]
            p2 = probs[..., m:]

            b1 = self.buckets[:m]
            b2 = self.buckets[m:]
            v = (torch.flip(p1 * b1, dims=[-1]) + (p2 * b2)).sum(-1)

        if return_logits:
            return v, logits
        else:
            return v

    def two_hot_loss(self, predicted_logits, targets, reduction="mean"):
        assert len(predicted_logits) == len(targets)

        indices = torch.bucketize(targets.contiguous(), self.buckets)
        indices = indices.clamp(1, len(self.buckets) - 1)

        left_returns = self.buckets[indices - 1]
        right_returns = self.buckets[indices]

        two_hot_targets = torch.zeros_like(predicted_logits)

        right_prob = (targets - left_returns) / (right_returns - left_returns)
        right_prob = right_prob.clamp(0.0, 1.0)
        right_prob = right_prob.to(two_hot_targets.dtype)

        indices = indices.unsqueeze(-1)
        right_prob = right_prob.unsqueeze(-1)

        two_hot_targets.scatter_(2, indices, right_prob)
        two_hot_targets.scatter_(2, indices - 1, 1.0 - right_prob)

        # Not sure how CE it behaves with extra dimensions, so flattening
        if two_hot_targets.dim() > 2:
            two_hot_targets = two_hot_targets.movedim(-1, 1)
        if predicted_logits.dim() > 2:
            predicted_logits = predicted_logits.movedim(-1, 1)

        # Currently works fine and the check might be causing GPU sync so commenting out for speed.
        # check that two hot targets are valid probs
        # assert torch.allclose(
        #     two_hot_targets.sum(dim=1), torch.ones_like(two_hot_targets.sum(dim=1))
        # ), "Two-hot targets do not sum to 1."
        # assert torch.all(
        #     (two_hot_targets >= 0.0) & (two_hot_targets <= 1.0)
        # ), "Two-hot targets have invalid probabilities."

        log_probs = self._ce(predicted_logits, two_hot_targets.detach())
        if reduction == "mean":
            log_probs = log_probs.mean()
        return log_probs

    def loss(self, predicted_logits, targets, reduction="mean"):
        return self.two_hot_loss(predicted_logits, targets, reduction=reduction)


class ActionHead(nn.Module):
    """
    Used for the Actor.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        action_dim=5,
        unimix_ratio=0.0,
        num_hidden_layers=3,
        fan: str = "in",
        outscale: float = 1.0,
        norm: str = "layernorm",
    ):
        super().__init__()
        self.backbone = MLP(input_dim, hidden_dim, num_layers=num_hidden_layers, fan=fan, outscale=1.0, norm=norm)
        self.head = nn.Linear(hidden_dim, action_dim)
        self.unimix_ratio = unimix_ratio
        self.action_dim = action_dim

        # Apply trunc_normal_init to final head layer with specified outscale
        trunc_normal_init(self.head.weight, fan=fan, scale=outscale)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        logits = self.head(self.backbone(x))

        # It doesn't matter whether we apply unimix during eval or not as long as the eval is greedy.
        # unimix just adds constant to every element so the argmax doesn't change.
        # However, eval might not be greedy (i. e. we need stochastic policy, that it's question what to do?)
        # REMOVING FOR NOW: I'd prefer to this in Dreamer
        # for ST we need probs and currently we do logits -> probs -> unimix -> probs -> logits -> probs -> ST
        # which is stupid. Instead of that we can do just logits -> probs -> unimix -> -> probs  ST
        # if self.unimix_ratio > 0.0 and self.training:
        #     p = unimix(logits, self.unimix_ratio)
        #     logits = torch.log(p)

        return logits

    def policy(self, x, add_unimix=False):
        logits = self.forward(x)
        if add_unimix:
            p = unimix(logits, self.unimix_ratio)
        else:
            p = torch.softmax(logits, dim=-1)
        return p


class RMSNormChannels(nn.RMSNorm):
    def __init__(self, features, eps=1e-4, dtype=torch.float32):
        # R2 dreamer explicitly uses torch.float32 for all RMSNorm computations, so we do the same here.
        super().__init__(features, eps=eps, dtype=dtype)

    def forward(self, x):
        # Assuming B, C, H, W
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)
