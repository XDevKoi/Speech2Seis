"""SeisBench extension for the Speech2Seis phase-picking model (``S2S_dpk``).

This module exposes :class:`S2S` -- a :class:`seisbench.models.base.WaveformModel`
wrapper around the shared Speech2Seis backbone and
:class:`~s2s.heads.HeadDetectionPicking` -- so the model can be used with the
standard SeisBench API::

    from s2s_dpk import S2S_dpk
    model = S2S_dpk(pretrained_path="/path/to/wav2vec2")
    model.load_state_dict(torch.load("model.pth")["model_dict"])
    annotations = model.annotate(stream)
    picks = model.classify(stream, P_threshold=0.3, S_threshold=0.3)

Input convention (same as the training model): a stream whose 3 components are
in ``[Z, N, E]`` order at 100 Hz; each trace is demeaned and std-normalised
inside ``annotate_batch_pre``.

The sub-module layout (``convs`` / ``llm_blocks`` / ``out_head``) matches the
reference training implementation, so checkpoints load without key remapping.

To register it inside ``seisbench.models``, copy this file into
``seisbench/models/`` (the ``s2s`` package must be importable) and add
``from .s2s_dpk import S2S_dpk`` to ``seisbench/models/__init__.py``.

Run ``python extras/seisbench/s2s_dpk.py`` from the repository root for a
single-sample demo (random weights).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import seisbench.util as sbu

from seisbench.models.base import WaveformModel

# Allow running this file directly from extras/seisbench/ inside the repo.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isdir(os.path.join(_REPO_ROOT, "s2s")) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from s2s.backbone import LLM_Block, Multi_Scale_Conv_Block
from s2s.common import lora_setting
from s2s.heads import HeadDetectionPicking

__all__ = ["S2S", "S2S_dpk"]

IN_SAMPLES = 6000


class S2S(WaveformModel):
    """Speech2Seis phase-picking model for SeisBench.

    Output channels are named ``S2S_N`` (noise), ``S2S_P`` and ``S2S_S``.
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (0, 0),
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 3000)

    def __init__(
        self,
        in_channels: int = 3,
        conv_scale_num: int = 4,
        conv_scale_strides=(8, 6, 4, 2),
        conv_channels=(16, 48, 96),
        conv_kernel_sizes=(16, 8, 6, 1),
        conv_strides=(2, 2, 1, 1),
        llm_layers: int = 3,
        d_model: int = 768,
        patch_size: int = 8,
        dp_head_channels=(128, 160, 192, 224),
        path_drop_rate: float = 0.3,
        mlp_drop_rate: float = 0.3,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        phases: str = "NPS",
        norm: str = "std",
        sampling_rate: int = 100,
        pretrain: bool = True,
        freeze: bool = True,
        pretrained_path: str | None = None,
        lora_config=None,
        **kwargs,
    ):
        citation = "Speech2Seis: seismic monitoring via cross-modal transfer with a pretrained speech model."
        super().__init__(
            citation=citation,
            in_samples=IN_SAMPLES,
            output_type="array",
            pred_sample=(0, IN_SAMPLES),
            labels=phases,
            sampling_rate=sampling_rate,
        )

        conv_channels = list(conv_channels)
        assert len(conv_channels) + 1 == len(conv_kernel_sizes) == len(conv_strides)
        conv_channels.append(d_model // patch_size)

        self.in_channels = in_channels
        self.norm = norm
        self.patch_size = patch_size
        self.feature_channels = conv_channels[-1]

        # --- Multi-scale convolutional embedder ---
        self.convs = nn.Sequential(
            *[
                Multi_Scale_Conv_Block(
                    scale_num=conv_scale_num,
                    scale_stride=scale_stride,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for scale_stride, in_dim, out_dim, kernel_size, stride in zip(
                    conv_scale_strides,
                    [in_channels] + conv_channels[:-1],
                    conv_channels,
                    conv_kernel_sizes,
                    conv_strides,
                )
            ]
        )

        # --- Pretrained speech encoder (LoRA fine-tuning) ---
        self.llm_blocks = LLM_Block(
            patch_size=patch_size,
            lora_config=lora_config or lora_setting(),
            pretrained_path=pretrained_path,
            pretrain=pretrain,
            freeze=freeze,
        )

        # --- Phase-picking decoder head ---
        out_layer_channels, out_layer_kernel_sizes = [], []
        for channel, kernel in zip(dp_head_channels, conv_kernel_sizes):
            out_layer_channels.insert(0, channel)
            out_layer_kernel_sizes.insert(0, kernel)

        self.out_head = HeadDetectionPicking(
            in_channels=in_channels,
            feature_channels=self.feature_channels,
            layer_channels=out_layer_channels,
            layer_kernel_sizes=out_layer_kernel_sizes,
            act_layer=act_layer,
            norm_layer=norm_layer,
            out_act_layer=nn.Sigmoid,
            out_channels=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x
        x = self.convs(x)
        x = self.llm_blocks(x)
        return self.out_head(x, x_input)

    # ------------------------------------------------------------------ #
    # SeisBench annotation hooks
    #
    # SeisBench >= 0.10 calls ``annotate_batch_pre`` / ``annotate_batch_post``;
    # older versions used ``annotate_window_pre`` / ``annotate_window_post``.
    # Both pairs are provided so the model works across SeisBench versions.
    # ------------------------------------------------------------------ #
    def _normalize(self, batch: torch.Tensor) -> torch.Tensor:
        norm = (self.norm or "std").lower()
        if norm == "std":
            batch = batch - batch.mean(dim=-1, keepdim=True)
            return batch / (batch.std(dim=-1, keepdim=True) + 1e-10)
        if norm == "mean_std":
            batch = batch - batch.mean(dim=-1, keepdim=True)
            std = batch.std(dim=-1, keepdim=True)
            return batch / (std.mean(dim=1, keepdim=True) + 1e-10)
        raise ValueError(f"Unsupported normalization method: {self.norm}")

    # --- modern API: batch shape (B, C, L) ---
    def annotate_batch_pre(self, batch: torch.Tensor, argdict: dict[str, Any]) -> torch.Tensor:
        """Normalize each trace according to ``self.norm``."""
        return self._normalize(batch)

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        # (B, C, L) -> (B, L, C): one channel per phase, as SeisBench expects.
        batch = batch.transpose(1, 2)
        prenan, postnan = argdict.get("blinding", self._annotate_args.get("blinding")[1])
        if prenan > 0:
            batch[:, :prenan] = float("nan")
        if postnan > 0:
            batch[:, -postnan:] = float("nan")
        return batch

    # --- legacy API: window shape (C, L) ---
    def annotate_window_pre(self, batch: torch.Tensor, argdict: dict[str, Any]) -> torch.Tensor:
        return self._normalize(batch)

    def annotate_window_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch.T
        prenan, postnan = argdict.get("blinding", self._annotate_args.get("blinding")[1])
        if prenan > 0:
            batch[:, :prenan] = np.nan
        if postnan > 0:
            batch[:, -postnan:] = np.nan
        return batch

    def classify_aggregate(self, annotations, argdict):
        """Convert probability curves into discrete picks using per-phase thresholds."""
        picks = sbu.PickList()
        for phase in self.labels:
            if phase == "N":
                continue
            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]),
                phase,
            )
        return sbu.ClassifyOutput(self.name, picks=sbu.PickList(sorted(picks)))

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
            "sampling_rate",
        ]:
            model_args.pop(key, None)
        model_args["in_channels"] = self.in_channels
        model_args["classes"] = getattr(self, "classes", list(self.labels))
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate
        return model_args


def S2S_dpk(**kwargs) -> S2S:
    """Create the Speech2Seis phase-picking model (:class:`S2S`)."""
    model = S2S(**kwargs)
    return model


if __name__ == "__main__":
    from obspy import Stream, Trace, UTCDateTime

    print("=" * 78)
    print("[WARNING] No pretrained backbone / task checkpoint is loaded.")
    print("          Using randomly initialised weights (`pretrain=False`) --")
    print("          the picks below are meaningless, this is only an API check.")
    print("=" * 78)

    # For real inference:
    #   model = S2S_dpk(pretrained_path="/path/to/wav2vec2")   # pretrain=True
    #   model.load_state_dict(torch.load("S2S_dpk.pth", map_location="cpu")["model_dict"])
    model = S2S_dpk(pretrain=False, freeze=False, in_channels=3)
    model.eval()

    rng = np.random.default_rng(0)
    start = UTCDateTime("2024-01-01T00:00:00")
    stream = Stream()
    for component in ("Z", "N", "E"):  # input component order must be [Z, N, E]
        stream += Trace(
            data=rng.standard_normal(IN_SAMPLES).astype(np.float32),
            header={
                "station": "S2S",
                "channel": f"BH{component}",
                "starttime": start,
                "sampling_rate": 100.0,
            },
        )

    with torch.no_grad():
        output = model.classify(stream, P_threshold=0.3, S_threshold=0.3)

    print(f"\nnumber of picks: {len(output.picks)}")
    for pick in output.picks[:10]:
        print(f"  {pick.phase}: {pick.peak_time}  ({pick.peak_value:.3f})")
