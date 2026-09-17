"""Speech2Seis (S2S) downstream task models.

Every task shares the :class:`~s2s.backbone.Speech2Seis` backbone and only swaps
the output head. The five public models are:

==================  ==========================  ==========================================
Model               Head                        Output
==================  ==========================  ==========================================
``S2S_dpk``         HeadDetectionPicking        ``(B, 3, L)`` sigmoid ``[N, P, S]``
``S2S_pmp``         HeadClassificationLocal     ``(B, 2)`` softmax ``[up, down]``
``S2S_baz``         HeadBAZLocal                ``(cos, sin)`` each ``(B, 1)``
``S2S_dis``         HeadRegressionLocal         ``(B, 1)`` kilometres
``S2S_bazdis``      HeadBAZDisLocal             ``(cos, sin, dis)``
==================  ==========================  ==========================================

See :data:`TASK_SPECS` for the exact input/output contract and loss of each task.
"""

from __future__ import annotations

from functools import partial
import os

import torch
import torch.nn as nn

from .backbone import Speech2Seis
from .common import ScaledActivation
from .heads import (
    HeadBAZDisLocal,
    HeadBAZLocal,
    HeadClassificationLocal,
    HeadDetectionPicking,
    HeadRegressionLocal,
)

__all__ = [
    "S2S_dpk",
    "S2S_pmp",
    "S2S_baz",
    "S2S_dis",
    "S2S_bazdis",
    "MODELS",
    "TASK_SPECS",
    "build_model",
    "load_checkpoint",
]

# Common defaults for the STEAD setting.
IN_SAMPLES = 6000
P_POSITION_RATIO = 0.5
LOCAL_HALF_WIDTH = 128

#: Short description of the input/output contract and training objective of
#: each task. Handy for documentation and for programmatic introspection.
TASK_SPECS = {
    "S2S_dpk": {
        "description": "Phase picking (P / S).",
        "input": "(B, 3, 6000) float32, channels [Z, N, E], demeaned and std-normalised per trace",
        "output": "(B, 3, 6000) sigmoid probabilities, channels [N, P, S]",
        "loss": "weighted BCE on [N, P, S] with weights [[0], [1], [1]]",
        "decode": "pick peaks of the P / S curves above a threshold (e.g. 0.3)",
        "centered": False,
    },
    "S2S_pmp": {
        "description": "P-motion polarity classification.",
        "input": "(B, 3, 6000), channels [Z, N, E] demeaned + std-normalised; P-centered (P at sample 3000)",
        "output": "(B, 2) softmax probabilities, channels [up, down]",
        "loss": "cross entropy with weights [1, 1]",
        "decode": "argmax over the two classes (0 = up, 1 = down)",
        "centered": True,
    },
    "S2S_baz": {
        "description": "Back-azimuth estimation.",
        "input": "(B, 3, 6000), channels [Z, N, E] demeaned + std-normalised; P-centered",
        "output": "(cos, sin), each (B, 1), from a Tanh activation",
        "loss": "Huber on the (cos, sin) targets (targets = baz in degrees)",
        "decode": "baz_deg = atan2(sin, cos) * 180 / pi, modulo 360",
        "centered": True,
    },
    "S2S_dis": {
        "description": "Epicentral distance estimation.",
        "input": "(B, 3, 6000), channels [Z, N, E] demeaned + std-normalised; P-centered",
        "output": "(B, 1) distance in kilometres (sigmoid * 500)",
        "loss": "Huber loss",
        "decode": "direct value in km",
        "centered": True,
    },
    "S2S_bazdis": {
        "description": "Joint back-azimuth + epicentral distance estimation.",
        "input": "(B, 3, 6000), channels [Z, N, E] demeaned + std-normalised; P-centered",
        "output": "(cos, sin, dis): two (B, 1) from Tanh + one (B, 1) km",
        "loss": "BAZDisLoss: w_baz * (Huber(cos) + Huber(sin)) + w_dis * Huber(dis)",
        "decode": "baz_deg = atan2(sin, cos) * 180 / pi; dis = km",
        "centered": True,
    },
}


def _build(output_head, pretrained_path=None, pretrain=True, freeze=True, **kwargs):
    return Speech2Seis(
        output_head=output_head,
        pretrained_path=pretrained_path,
        pretrain=pretrain,
        freeze=freeze,
        **kwargs,
    )


def S2S_dpk(pretrained_path=None, pretrain=True, freeze=True, **kwargs):
    """Detection and phase picking (dense ``[N, P, S]`` curves)."""
    return _build(
        partial(HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3),
        pretrained_path=pretrained_path,
        pretrain=pretrain,
        freeze=freeze,
        path_drop_rate=0.3,
        mlp_drop_rate=0.3,
        **kwargs,
    )


def S2S_pmp(pretrained_path=None, pretrain=True, freeze=True, **kwargs):
    """P-motion polarity classification (P-centered, output ``(B, 2)``)."""
    return _build(
        partial(
            HeadClassificationLocal,
            out_act_layer=partial(nn.Softmax, dim=-1),
            num_classes=2,
            p_position_ratio=P_POSITION_RATIO,
            local_half_width=LOCAL_HALF_WIDTH,
        ),
        pretrained_path=pretrained_path,
        pretrain=pretrain,
        freeze=freeze,
        path_drop_rate=0.3,
        mlp_drop_rate=0.3,
        **kwargs,
    )


def S2S_baz(pretrained_path=None, pretrain=True, freeze=True, **kwargs):
    """Back-azimuth estimation (P-centered, output ``(cos, sin)``)."""
    return _build(
        partial(
            HeadBAZLocal,
            out_act_layer=nn.Tanh,
            p_position_ratio=P_POSITION_RATIO,
            local_half_width=LOCAL_HALF_WIDTH,
        ),
        pretrained_path=pretrained_path,
        pretrain=pretrain,
        freeze=freeze,
        path_drop_rate=0.3,
        mlp_drop_rate=0.3,
        **kwargs,
    )


def S2S_dis(pretrained_path=None, pretrain=True, freeze=True, **kwargs):
    """Epicentral distance estimation (P-centered, output km)."""
    return _build(
        partial(
            HeadRegressionLocal,
            out_act_layer=partial(ScaledActivation, act_layer=nn.Sigmoid, scale_factor=500),
            p_position_ratio=P_POSITION_RATIO,
            local_half_width=LOCAL_HALF_WIDTH,
        ),
        pretrained_path=pretrained_path,
        pretrain=pretrain,
        freeze=freeze,
        path_drop_rate=0.3,
        mlp_drop_rate=0.3,
        **kwargs,
    )


def S2S_bazdis(pretrained_path=None, pretrain=True, freeze=True, **kwargs):
    """Joint back-azimuth + distance estimation (P-centered, output 3-tuple)."""
    return _build(
        partial(
            HeadBAZDisLocal,
            out_act_layer_baz=nn.Tanh,
            out_act_layer_dis=partial(ScaledActivation, act_layer=nn.Sigmoid, scale_factor=500),
            p_position_ratio=P_POSITION_RATIO,
            local_half_width=LOCAL_HALF_WIDTH,
        ),
        pretrained_path=pretrained_path,
        pretrain=pretrain,
        freeze=freeze,
        path_drop_rate=0.3,
        mlp_drop_rate=0.3,
        **kwargs,
    )


#: Name -> factory function.
MODELS = {
    "S2S_dpk": S2S_dpk,
    "S2S_pmp": S2S_pmp,
    "S2S_baz": S2S_baz,
    "S2S_dis": S2S_dis,
    "S2S_bazdis": S2S_bazdis,
}


def build_model(name: str, **kwargs) -> Speech2Seis:
    """Instantiate a Speech2Seis task model by name.

    Args:
        name: one of :data:`MODELS`.
        **kwargs: forwarded to the factory (e.g. ``pretrained_path``,
            ``pretrain``, ``freeze``, ``in_channels``).

    Example:
        >>> model = build_model("S2S_dpk", pretrained_path="/path/to/wav2vec2")
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name](**kwargs)


def load_checkpoint(model: nn.Module, checkpoint: str, map_location="cpu", strict: bool = False):
    """Load a trained Speech2Seis checkpoint into ``model``.

    Accepts both raw ``state_dict`` files and the training checkpoints that wrap
    it as ``{"model_dict": ...}``, and strips ``module.`` / ``_orig_mod.``
    prefixes produced by DDP / ``torch.compile``.
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)

    ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("model_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return missing, unexpected
