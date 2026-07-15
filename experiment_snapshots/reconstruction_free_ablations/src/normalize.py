import torch
import torch.nn as nn


class PercScaler(nn.Module):
    def __init__(self, lo_p, hi_p, ema_decay=0.99):
        super().__init__()

        self.lo_p = lo_p
        self.hi_p = hi_p
        self.ema_decay = ema_decay
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(0.0))

    def update_averages(self, x):
        lo = torch.quantile(x, self.lo_p)
        hi = torch.quantile(x, self.hi_p)
        self.lo = self.ema_decay * self.lo + (1 - self.ema_decay) * lo.detach()
        self.hi = self.ema_decay * self.hi + (1 - self.ema_decay) * hi.detach()

    def scale(self, x):
        ret_s = (self.hi - self.lo).clamp(min=1.0).detach()
        return x / ret_s

    def forward(self, to_scale, for_stats):
        self.update_averages(for_stats)
        return self.scale(to_scale)
