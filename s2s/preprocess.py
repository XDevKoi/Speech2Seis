"""Inference-time waveform preprocessing for Speech2Seis.

These helpers reproduce the preprocessing used during training:

* :func:`normalize` -- per-component demeaning + standard-deviation scaling,
* :func:`cut_window_p_centered` -- place the P arrival at
  ``p_position_ratio * in_samples`` and zero-pad out-of-range samples.

The P-centered models (``S2S_pmp``, ``S2S_baz``, ``S2S_dis``, ``S2S_bazdis``)
**require** the P-centered window; ``S2S_dpk`` works on any window of
``in_samples`` samples.
"""

from __future__ import annotations

import numpy as np

__all__ = ["normalize", "cut_window_p_centered", "prepare_p_centered", "fit_length"]

IN_SAMPLES = 6000
P_POSITION_RATIO = 0.5


def normalize(data: np.ndarray, mode: str = "std") -> np.ndarray:
    """Normalize each component of ``data`` of shape ``(C, L)``.

    Args:
        data: 2-D array, one row per component.
        mode: ``"std"`` (demean + divide by std), ``"max"`` (demean + divide by
            max abs), ``"mean_std"`` (demean + divide by the mean channel std),
            or ``""`` / ``"none"``.
    """
    data = data.copy()
    data -= np.mean(data, axis=1, keepdims=True)
    if mode == "max":
        scale = np.max(np.abs(data), axis=1, keepdims=True)
        scale[scale == 0] = 1
    elif mode == "std":
        scale = np.std(data, axis=1, keepdims=True)
        scale[scale == 0] = 1
    elif mode == "mean_std":
        scale = np.mean(np.std(data, axis=1, keepdims=True))
        scale = 1 if scale == 0 else scale
    elif mode in ("", "none"):
        return data
    else:
        raise ValueError(f"Supported modes: 'max', 'std', 'mean_std', 'none', got '{mode}'")
    return data / scale


def cut_window_p_centered(
    data: np.ndarray,
    p_idx: int,
    in_samples: int = IN_SAMPLES,
    p_position_ratio: float = P_POSITION_RATIO,
) -> np.ndarray:
    """Cut a ``(C, in_samples)`` window with the P arrival at ``p_position_ratio``.

    Samples outside the waveform are zero-padded.
    """
    window = np.zeros((data.shape[0], in_samples), dtype=np.float32)
    shift = int(in_samples * p_position_ratio)
    c_l = p_idx - shift
    c_r = c_l + in_samples

    s_l, s_r = max(c_l, 0), min(c_r, data.shape[-1])
    t_l = max(0, -c_l)
    t_r = t_l + (s_r - s_l)
    if s_r > s_l:
        window[:, t_l:t_r] = data[:, s_l:s_r]
    return window


def prepare_p_centered(
    data: np.ndarray,
    p_idx: int,
    in_samples: int = IN_SAMPLES,
    p_position_ratio: float = P_POSITION_RATIO,
    norm_mode: str = "std",
    normalize_window: bool = False,
) -> np.ndarray:
    """Full preprocessing for the P-centered tasks.

    Args:
        data: waveform ``(C, L)`` in model channel order (``[Z, N, E]``).
        p_idx: index of the P arrival in ``data``.
        normalize_window: if ``True``, normalize again *after* cutting (used by
            ``S2S_pmp``); the regression tasks only normalize the full trace.
    """
    data = normalize(data, norm_mode)
    window = cut_window_p_centered(data, p_idx, in_samples, p_position_ratio)
    if normalize_window:
        window = normalize(window, norm_mode)
    return window


def fit_length(data: np.ndarray, in_samples: int = IN_SAMPLES) -> np.ndarray:
    """Crop (from the start) or zero-pad a waveform to ``in_samples`` samples."""
    length = data.shape[-1]
    if length >= in_samples:
        return data[:, :in_samples].astype(np.float32)
    return np.concatenate(
        [data, np.zeros((data.shape[0], in_samples - length), dtype=data.dtype)], axis=1
    ).astype(np.float32)
