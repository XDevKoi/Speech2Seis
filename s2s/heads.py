"""Task heads for Speech2Seis (S2S).

All heads consume the shared backbone feature map ``x`` of shape ``(B, C, L')``
with ``C = 96`` and ``L' = in_samples / prod(conv_strides)`` (``L' = 1500`` for
the standard 6000-sample STEAD configuration).

Heads come in two flavours:

* **Dense / decoder** -- :class:`HeadDetectionPicking` upsamples the feature map
  back to the input length and produces per-sample curves (phase picking).
* **P-centered local** -- :class:`HeadClassificationLocal`,
  :class:`HeadRegressionLocal`, :class:`HeadBAZLocal` and
  :class:`HeadBAZDisLocal` crop a fixed window around the P arrival before
  pooling. They require the input window to be P-centered, i.e. P must sit at
  ``p_position_ratio * in_samples`` (default ``0.5``) after preprocessing.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _auto_pad_1d


class HeadDetectionPicking(nn.Module):
    """Dense decoder head for detection / phase picking.

    Progressively upsamples ``(B, C, L')`` back to ``(B, out_channels, L)``
    (``L`` = input waveform length) using linear interpolation + Conv1d blocks.

    For ``out_channels=3`` and ``Sigmoid`` output activation the channels are
    ``[N, P, S]`` (noise / P / S).
    """

    def __init__(
        self,
        feature_channels: int,
        layer_channels: list[int],
        layer_kernel_sizes: list[int],
        act_layer,
        norm_layer,
        out_act_layer=nn.Identity,
        out_channels: int = 1,
        **kwargs,
    ):
        super().__init__()

        assert len(layer_channels) == len(layer_kernel_sizes)

        self.depth = len(layer_channels)
        self.up_layers = nn.ModuleList()

        for inc, outc, kers in zip(
            [feature_channels] + layer_channels[:-1],
            layer_channels[:-1] + [out_channels * 2],
            layer_kernel_sizes,
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers)
            norm = norm_layer(outc)
            act = act_layer()
            self.up_layers.append(
                nn.Sequential(
                    OrderedDict([("conv", conv), ("norm", norm), ("act", act)])
                )
            )

        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
        )
        self.out_act = out_act_layer()

    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        _, _, length = x.size()
        up_sizes = self._upsampling_sizes(in_size=length, out_size=x0.size(-1))
        for i, layer in enumerate(self.up_layers):
            x = F.interpolate(x, size=up_sizes[i], mode="linear")
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x


class HeadClassificationLocal(nn.Module):
    """Classification head using LOCAL features around the P arrival.

    The input window is expected to be P-centered. After the conv embedder
    (downsample factor ``prod(conv_strides) = 4``), P corresponds to feature
    index ``p_position_ratio * L'``; only a window of ``local_half_width``
    features on each side is pooled, so the polarity signal is not averaged out
    by the rest of the (long) trace.

    Used for ``S2S_pmp``. Output: ``(B, num_classes)``.
    """

    def __init__(
        self,
        feature_channels: int,
        num_classes: int,
        out_act_layer,
        p_position_ratio: float = 0.5,
        local_half_width: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.p_position_ratio = p_position_ratio
        self.local_half_width = local_half_width
        self.convs = nn.ModuleList(
            [nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels, num_classes)
        self.out_act = out_act_layer()

    def forward(self, x: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        length = x.size(-1)
        center = int(round(self.p_position_ratio * length))
        half = self.local_half_width
        lo, hi = max(0, center - half), min(length, center + half)
        x = x[:, :, lo:hi]
        if x.size(-1) < 16:
            x = F.pad(x, (0, 16 - x.size(-1)))
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class HeadRegressionLocal(nn.Module):
    """Scalar regression head using LOCAL features around the P arrival.

    Same P-centered local pooling as :class:`HeadClassificationLocal`.
    Used for ``S2S_dis`` (distance in km) and the distance branch of
    ``S2S_bazdis``. Output: ``(B, 1)``.
    """

    def __init__(
        self,
        feature_channels: int,
        out_act_layer,
        p_position_ratio: float = 0.5,
        local_half_width: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.p_position_ratio = p_position_ratio
        self.local_half_width = local_half_width
        self.convs = nn.ModuleList(
            [nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels, 1)
        self.out_act = out_act_layer()

    def forward(self, x: torch.Tensor, _: torch.Tensor = None) -> torch.Tensor:
        length = x.size(-1)
        center = int(round(self.p_position_ratio * length))
        half = self.local_half_width
        lo, hi = max(0, center - half), min(length, center + half)
        x = x[:, :, lo:hi]
        if x.size(-1) < 16:
            x = F.pad(x, (0, 16 - x.size(-1)))
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class HeadBAZLocal(nn.Module):
    """Back-azimuth head using LOCAL features around the P arrival.

    Outputs the ``(cos, sin)`` representation of the back-azimuth as a tuple of
    two ``(B, 1)`` tensors. Decode with ``atan2(sin, cos)``.
    Used for ``S2S_baz``.
    """

    def __init__(
        self,
        feature_channels: int,
        out_act_layer,
        p_position_ratio: float = 0.5,
        local_half_width: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.p_position_ratio = p_position_ratio
        self.local_half_width = local_half_width
        self.convs = nn.ModuleList(
            [nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels, 2)
        self.out_act = out_act_layer()

    def forward(self, x: torch.Tensor, _: torch.Tensor = None):
        length = x.size(-1)
        center = int(round(self.p_position_ratio * length))
        half = self.local_half_width
        lo, hi = max(0, center - half), min(length, center + half)
        x = x[:, :, lo:hi]
        if x.size(-1) < 16:
            x = F.pad(x, (0, 16 - x.size(-1)))
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x[:, :1], x[:, 1:]


class HeadBAZDisLocal(nn.Module):
    """Dual-branch head for joint back-azimuth + epicentral-distance estimation.

    Same P-centered local feature pooling as :class:`HeadBAZLocal` /
    :class:`HeadRegressionLocal`, split into two independent branches:

    * baz branch -> ``(cos, sin)`` via ``out_act_layer_baz`` (Tanh),
    * dis branch -> scalar kilometres via ``out_act_layer_dis``
      (sigmoid * scale_factor).

    Returns the 3-tuple ``(cos, sin, dis)``. Used for ``S2S_bazdis``.
    """

    def __init__(
        self,
        feature_channels: int,
        out_act_layer_baz,
        out_act_layer_dis,
        p_position_ratio: float = 0.5,
        local_half_width: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.p_position_ratio = p_position_ratio
        self.local_half_width = local_half_width

        self.baz_convs = nn.ModuleList(
            [nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)]
        )
        self.baz_pool = nn.AdaptiveAvgPool1d(1)
        self.baz_lin = nn.Linear(feature_channels, 2)
        self.baz_out_act = out_act_layer_baz()

        self.dis_convs = nn.ModuleList(
            [nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)]
        )
        self.dis_pool = nn.AdaptiveAvgPool1d(1)
        self.dis_lin = nn.Linear(feature_channels, 1)
        self.dis_out_act = out_act_layer_dis()

    def forward(self, x: torch.Tensor, _: torch.Tensor = None):
        length = x.size(-1)
        center = int(round(self.p_position_ratio * length))
        half = self.local_half_width
        lo, hi = max(0, center - half), min(length, center + half)
        x = x[:, :, lo:hi]
        if x.size(-1) < 16:
            x = F.pad(x, (0, 16 - x.size(-1)))

        xb = x
        for conv in self.baz_convs:
            xb = conv(xb)
        xb = self.baz_pool(xb)
        xb = self.baz_lin(xb.flatten(1))
        xb = self.baz_out_act(xb)

        xd = x
        for conv in self.dis_convs:
            xd = conv(xd)
        xd = self.dis_pool(xd)
        xd = self.dis_lin(xd.flatten(1))
        xd = self.dis_out_act(xd)

        return xb[:, :1], xb[:, 1:], xd
