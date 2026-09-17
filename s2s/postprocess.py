"""Post-processing helpers for Speech2Seis predictions.

Currently provides the peak picking used to turn the ``S2S_dpk`` probability
curves into discrete P / S arrival samples. The logic mirrors the reference
evaluation post-processing (threshold + minimum peak distance).
"""

from __future__ import annotations

import numpy as np

__all__ = ["peaks_above_threshold", "decode_dpk"]


def peaks_above_threshold(signal, threshold: float = 0.3, min_distance: int = 50):
    """Find local maxima above a threshold.

    Args:
        signal: 1-D probability curve of shape ``(L,)``.
        threshold: probability threshold.
        min_distance: minimum separation (in samples) between two picks.

    Returns:
        List of ``[sample_index, probability]`` pairs, sorted by index.
    """
    signal = np.asarray(signal)
    above = signal > threshold
    peaks: list[list[float]] = []
    i, length = 0, len(above)
    while i < length:
        if above[i]:
            start = i
            while i + 1 < length and above[i + 1]:
                i += 1
            end = i + 1
            segment = signal[start:end]
            local = int(np.argmax(segment))
            peak_idx, peak_val = start + local, float(segment[local])
            if not peaks or peak_idx - peaks[-1][0] >= min_distance:
                peaks.append([peak_idx, peak_val])
            elif peak_val > peaks[-1][1]:
                peaks[-1] = [peak_idx, peak_val]
        i += 1
    return peaks


def decode_dpk(probs, threshold: float = 0.3, min_distance: int = 50):
    """Decode ``S2S_dpk`` output into P / S picks.

    Args:
        probs: sigmoid probabilities ``(3, L)`` or ``(B, 3, L)`` with channel
            order ``[N, P, S]``.
        threshold: probability threshold for a pick.
        min_distance: minimum separation between picks of the same phase.

    Returns:
        ``{"P": [[sample, prob], ...], "S": [[sample, prob], ...]}``.
    """
    probs = np.asarray(probs)
    if probs.ndim == 3:
        probs = probs[0]
    return {
        "P": peaks_above_threshold(probs[1], threshold, min_distance),
        "S": peaks_above_threshold(probs[2], threshold, min_distance),
    }
