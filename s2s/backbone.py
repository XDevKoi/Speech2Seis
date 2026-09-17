"""Speech2Seis backbone.

The backbone transfers a pretrained speech model (Wav2Vec2) to seismic
waveforms through three components:

1. a **Multi-Scale Convolutional Embedder** that downsamples the 3-component
   waveform and maps it to ``d_model / patch_size`` feature channels,
2. a **pretrained Wav2Vec2 Transformer encoder** used as the sequence model
   (fine-tuned with LoRA; its internal CNN feature extractor is bypassed),
3. a pluggable **task head** (see :mod:`s2s.heads`).

The standard STEAD configuration uses ``in_samples=6000`` and
``conv_strides=[2, 2, 1, 1]`` (total downsample factor 4), giving a feature map
of shape ``(B, 96, 1500)`` which is patchified into ``187 x 768`` tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch._dynamo.config
from einops import rearrange

import transformers.models.wav2vec2 as Wav2Vec2
from peft import get_peft_model

from .common import (
    DEFAULT_WAV2VEC2_PATH,
    _auto_pad_1d,
    lora_setting,
    resolve_wav2vec2_path,
)
from .heads import HeadDetectionPicking

torch._dynamo.config.cache_size_limit = 1024

__all__ = ["Speech2Seis", "LLM_Block", "Multi_Scale_Conv_Block", "ConvBlock"]


class ConvBlock(nn.Module):
    """1x1 channel projection + Conv1d + norm + activation."""

    def __init__(self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, bias=False)
        self.norm = norm_layer(out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.in_proj(x)
        x = _auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Multi_Scale_Conv_Block(nn.Module):
    """Parallel multi-scale convolution block.

    ``scale_num`` branches use kernel sizes ``kernel_size + scale_stride * i``
    for ``i`` in ``range(scale_num)``; the concatenated outputs are fused by a
    1x1 convolution.
    """

    def __init__(
        self,
        scale_num,
        scale_stride,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        act_layer,
        norm_layer,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                ConvBlock(
                    in_dim,
                    out_dim,
                    kernel_size + int(scale_stride * scale),
                    stride,
                    act_layer,
                    norm_layer,
                )
                for scale in range(scale_num)
            ]
        )
        self.out_proj = nn.Conv1d(scale_num * out_dim, out_dim, kernel_size=1, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.out_proj(x)
        x = self.norm(x)
        return x


class LLM_Block(nn.Module):
    """Pretrained Wav2Vec2 Transformer encoder with LoRA fine-tuning.

    The internal CNN ``feature_extractor`` of Wav2Vec2 is discarded; only the
    Transformer encoder is kept. The convolutional features are patchified with
    ``unfold(..., size=patch_size, step=patch_size)`` so that each patch is
    projected to ``d_model`` by concatenating ``patch_size`` feature channels.
    """

    def __init__(
        self,
        patch_size: int = 8,
        lora_config=None,
        pretrained_path: str | None = None,
        pretrain: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size

        if pretrain:
            full_model = Wav2Vec2.Wav2Vec2Model.from_pretrained(
                resolve_wav2vec2_path(pretrained_path),
                output_hidden_states=True,
                ignore_mismatched_sizes=True,
            )
            self.llm = full_model.encoder
        else:
            self.llm = Wav2Vec2.Wav2Vec2Model(Wav2Vec2.Wav2Vec2Config()).encoder

        self.llm.config.layerdrop = 0.0

        if freeze and pretrain:
            # Freeze the pretrained weights and train only the LoRA adapters and
            # the LayerNorm parameters.
            self.llm = get_peft_model(self.llm, lora_config or lora_setting())
            for name, param in self.llm.named_parameters():
                param.requires_grad = "layer_norm" in name or "lora" in name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, L') -> (B, L'/p, C*p)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        x = rearrange(x, "b c n p -> b n (c p)")
        x = self.llm(x).last_hidden_state
        # (B, L'/p, C*p) -> (B, C, n*p)
        x = rearrange(x, "b n (c p) -> b c (n p)", p=self.patch_size)
        return x


class Speech2Seis(nn.Module):
    """Speech2Seis backbone (encoder + task head).

    Args:
        in_channels: number of input components (3 for ``[Z, N, E]``).
        conv_scale_num: number of parallel scales in each conv block.
        conv_scale_strides: per-stage kernel stride for the multi-scale branches.
        conv_channels: output channels of each conv stage.
        conv_kernel_sizes: base kernel size of each conv stage.
        conv_strides: temporal stride of each conv stage.
        d_model: hidden size of the pretrained encoder (768 for Wav2Vec2-base).
        patch_size: temporal patch size fed to the encoder.
        dp_head_channels: channel widths for :class:`HeadDetectionPicking`.
        output_head: the task head (a class or ``functools.partial``).
        pretrain: load the pretrained speech weights.
        freeze: freeze the encoder and train LoRA + LayerNorm only.
        pretrained_path: local directory or HuggingFace id of Wav2Vec2.
        lora_config: optional custom :class:`peft.LoraConfig`.
    """

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
        path_drop_rate: float = 0.2,
        mlp_drop_rate: float = 0.2,
        mlp_ratio: int = 4,
        mlp_bias: bool = True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        use_checkpoint: bool = False,
        output_head=None,
        pretrain: bool = True,
        freeze: bool = True,
        pretrained_path: str | None = None,
        lora_config=None,
        **kwargs,
    ):
        super().__init__()

        conv_channels = list(conv_channels)
        assert len(conv_channels) + 1 == len(conv_kernel_sizes) == len(conv_strides)
        conv_channels.append(d_model // patch_size)

        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.feature_channels = conv_channels[-1]
        self.in_channels = in_channels

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

        # --- Pretrained speech encoder ---
        self.llm_blocks = LLM_Block(
            patch_size=patch_size,
            lora_config=lora_config,
            pretrained_path=pretrained_path,
            pretrain=pretrain,
            freeze=freeze,
        )

        # --- Task head ---
        if output_head is None:
            raise ValueError("`output_head` must be provided (see s2s.heads).")

        head_fn = getattr(output_head, "func", output_head)
        if head_fn is HeadDetectionPicking:
            out_layer_channels, out_layer_kernel_sizes = [], []
            for channel, kernel in zip(dp_head_channels, conv_kernel_sizes):
                out_layer_channels.insert(0, channel)
                out_layer_kernel_sizes.insert(0, kernel)
            self.out_head = output_head(
                in_channels=in_channels,
                feature_channels=self.feature_channels,
                layer_channels=out_layer_channels,
                layer_kernel_sizes=out_layer_kernel_sizes,
                act_layer=act_layer,
                norm_layer=norm_layer,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias,
            )
        else:
            self.out_head = output_head(
                feature_channels=self.feature_channels,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

    def forward(self, x: torch.Tensor):
        """Run the backbone + head.

        Args:
            x: waveform tensor ``(B, in_channels, in_samples)`` with channels in
                ``[Z, N, E]`` order, demeaned and std-normalised per trace.

        Returns:
            The task head output. Shape depends on the task, see
            :data:`s2s.models.TASK_SPECS`.
        """
        x_input = x
        x = self.convs(x)
        x = self.llm_blocks(x)
        return self.out_head(x, x_input)
