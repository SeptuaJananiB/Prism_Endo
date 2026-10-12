"""Minimal AdamW using only element-wise ops that DirectML implements.

torch.optim.AdamW calls `lerp_`, which the DirectML backend runs on the CPU,
copying every parameter's state off the GPU each step. This version avoids
that. Maths matches torch.optim.AdamW (decoupled weight decay, bias
correction, no amsgrad). Used automatically on DirectML devices only.
"""

from __future__ import annotations

import math

import torch


class SimpleAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                # int(): torch.optim.AdamW stores step as a tensor (resume from its state)
                st["step"] = int(st["step"]) + 1
                t = st["step"]
                if wd:
                    p.mul_(1 - lr * wd)
                st["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1 = 1 - b1 ** t
                bc2 = 1 - b2 ** t
                denom = (st["exp_avg_sq"] / bc2).sqrt_().add_(eps)
                p.addcdiv_(st["exp_avg"], denom, value=-lr / bc1)
        return loss
