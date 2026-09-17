"""Shared building blocks for Speech2Seis (S2S).

This module holds the small utilities used by both the backbone and the task
heads, so that neither has to import the other (avoids circular imports).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig

# The pretrained speech model that is transferred to seismology.
DEFAULT_WAV2VEC2_PATH = "facebook/wav2vec2-base-960h"

# Download address of the pretrained Speech2Seis backbone (Wav2Vec2).
WAV2VEC2_DOWNLOAD_URL = "https://huggingface.co/facebook/wav2vec2-base-960h/tree/main"

# Environment variables checked (in order) when no explicit path is given.
WAV2VEC2_PATH_ENV_VARS = ("S2S_WAV2VEC2_PATH", "WAV2VEC2_PATH")


def resolve_wav2vec2_path(path: str | None = None) -> str:
    """Resolve the local/HuggingFace path of the pretrained Wav2Vec2 model.

    Priority:
        1. explicit ``path`` argument,
        2. ``S2S_WAV2VEC2_PATH`` / ``WAV2VEC2_PATH`` environment variables,
        3. the HuggingFace hub id ``facebook/wav2vec2-base-960h``.
    """
    if path:
        return path
    for var in WAV2VEC2_PATH_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return DEFAULT_WAV2VEC2_PATH


def lora_setting(
    target_modules="all-linear",
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    bias: str = "lora_only",
) -> LoraConfig:
    """Default LoRA configuration used for parameter-efficient fine-tuning."""
    return LoraConfig(
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
    )


def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Pad ``x`` so a conv layer yields ``ceil(x.size(dim) / stride)`` outputs.

    Replaces ``padding='same'``, which TorchScript / ONNX do not support.
    """
    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    return F.pad(x, padding, "constant", padding_value)


class ScaledActivation(nn.Module):
    """Apply ``act_layer`` and rescale its output by a constant factor.

    Used by the distance heads to map a sigmoid output in ``[0, 1]`` to a
    physical range in kilometres, e.g. ``sigmoid(x) * 500``.
    """

    def __init__(self, act_layer: nn.Module, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) * self.scale_factor
