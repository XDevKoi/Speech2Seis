"""Training objectives for the Speech2Seis downstream tasks.

These mirror the losses used to train the released checkpoints. They are
provided so the task I/O in :data:`s2s.models.TASK_SPECS` is fully specified,
even though this repository focuses on model definitions and inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BCELoss", "CELoss", "BAZDisLoss"]


class BCELoss(nn.Module):
    """Weighted binary cross entropy for dense curves.

    Used by ``S2S_dpk`` with ``weight=[0, 1, 1]`` on ``[N, P, S]`` (the noise
    channel is ignored; P and S are optimised directly).
    """

    _epsilon = 1e-6

    def __init__(self, weight=None):
        super().__init__()
        if weight is None:
            weight = 1.0
        self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))

    def forward(self, preds, targets):
        """``preds`` and ``targets`` have shape ``(N, C, L)``."""
        loss_pos = -targets * torch.log(preds + self._epsilon)
        loss_neg = -(1 - targets) * torch.log(1 - preds + self._epsilon)
        return ((loss_pos + loss_neg) * self.weight).mean()


class CELoss(nn.Module):
    """Weighted cross entropy for classification.

    Used by ``S2S_pmp`` (``preds`` is already softmaxed, shape ``(N, C)``).
    """

    _epsilon = 1e-6

    def __init__(self, weight=None):
        super().__init__()
        if weight is None:
            weight = 1.0
        self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))

    def forward(self, preds, targets):
        loss = -targets * torch.log(preds + self._epsilon)
        return (loss * self.weight).sum(1).mean()


class BAZDisLoss(nn.Module):
    """Joint back-azimuth + epicentral-distance loss.

    ``preds`` / ``targets`` are 3-tuples ``(cos, sin, dis)``. The total is
    ``w_baz * (Huber(cos) + Huber(sin)) + w_dis * Huber(dis)``.
    """

    def __init__(self, w_baz: float = 0.95, w_dis: float = 0.05, delta: float = 1.0):
        super().__init__()
        self.w_baz = w_baz
        self.w_dis = w_dis
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, preds, targets):
        cos_p, sin_p, dis_p = preds
        cos_t, sin_t, dis_t = targets
        return (
            self.w_baz * (self.huber(cos_p, cos_t) + self.huber(sin_p, sin_t))
            + self.w_dis * self.huber(dis_p, dis_t)
        )
