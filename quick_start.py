"""Single-sample test for all five Speech2Seis downstream tasks.

The test sample is drawn from the **test set** (see ``extras/samples/``); if
several ``*.npz`` samples are present, one is picked at random.

Trained weights are discovered automatically in ``extras/checkpoints/``:

* if ``extras/checkpoints/<task>.pth`` exists, it is loaded (the pretrained
  Wav2Vec2 backbone is resolved through ``S2S_WAV2VEC2_PATH`` / the
  ``pretrained_path`` argument, otherwise it is fetched from HuggingFace);
* if no checkpoint is found, the model is built with **randomly initialised**
  weights and a warning is printed for that task.

Input contract (all tasks): a ``(1, 3, 6000)`` float32 tensor with channels in
``[Z, N, E]`` order, demeaned and std-normalised per trace. ``S2S_dpk`` is run on
the full trace; the P-centered tasks (``S2S_pmp`` / ``S2S_baz`` / ``S2S_dis`` /
``S2S_bazdis``) use a window with the P arrival at sample ``0.5 * 6000 = 3000``.

Run:
    python quick_start.py

For GPU inference set the device via the standard PyTorch mechanism, e.g.
``CUDA_VISIBLE_DEVICES=0``; this script runs on CPU by default.
"""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from s2s import (  # noqa: E402
    TASK_SPECS,
    build_model,
    decode_dpk,
    fit_length,
    load_checkpoint,
    normalize,
    prepare_p_centered,
)

IN_SAMPLES = 6000
P_POSITION_RATIO = 0.5
SAMPLE_DIR = os.path.join(REPO_ROOT, "extras", "samples")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "extras", "checkpoints")

# Optional: explicit local Wav2Vec2 directory (or set S2S_WAV2VEC2_PATH). Leave
# empty to let the library resolve it (env var, then HuggingFace hub id).
BACKBONE_PATH = ""


def load_test_sample(rng: np.random.Generator):
    """Return (waveform [Z,N,E], p_idx, label dict, name) of a random test sample.

    Falls back to a synthetic trace only if no sample file is available.
    """
    paths = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.npz")))
    if not paths:
        print(f"[WARNING] No test sample found in {SAMPLE_DIR}; using synthetic noise.\n")
        waveform = rng.standard_normal((3, 8000)).astype(np.float32)
        return waveform, 4000, {}, "<synthetic>"

    path = paths[int(rng.integers(len(paths)))]
    sample = np.load(path, allow_pickle=True)
    labels = {
        "source": str(sample["source"]) if "source" in sample else "unknown",
        "trace_id": str(sample["trace_id"]) if "trace_id" in sample else "?",
        "baz_deg": float(sample["baz_deg"]) if "baz_deg" in sample else None,
        "dis_km": float(sample["dis_km"]) if "dis_km" in sample else None,
        "pmp": int(sample["pmp"]) if "pmp" in sample else -1,
    }
    return sample["waveform"], int(sample["p_idx"]), labels, os.path.basename(path)


def build_for_task(task: str):
    """Build ``task``, loading ``extras/checkpoints/<task>.pth`` when present.

    Returns ``(model, status_string)``. When no checkpoint exists the model is
    built with randomly initialised weights.
    """
    checkpoint = os.path.join(CHECKPOINT_DIR, f"{task}.pth")
    if os.path.isfile(checkpoint):
        model = build_model(task, pretrained_path=BACKBONE_PATH or None)
        missing, unexpected = load_checkpoint(model, checkpoint)
        detail = "" if not (missing or unexpected) else (
            f" (missing={len(missing)}, unexpected={len(unexpected)})")
        status = f"loaded {os.path.relpath(checkpoint, REPO_ROOT)}{detail}"
    else:
        model = build_model(task, pretrain=False, freeze=False)
        status = "random init (no checkpoint in extras/checkpoints)"
    return model, status


def describe(task: str, out) -> None:
    """Print the decoded output of ``task`` for a single sample."""
    if task == "S2S_dpk":
        probs = out[0].cpu().numpy()  # (3, L): [N, P, S]
        print(f"    output: (1, 3, {probs.shape[-1]}) sigmoid probs [N, P, S], "
              f"mean = [{probs[0].mean():.3f}, {probs[1].mean():.3f}, {probs[2].mean():.3f}]")
        picks = decode_dpk(probs, threshold=0.3, min_distance=50)
        p_picks = [int(i) for i, _ in picks["P"]]
        s_picks = [int(i) for i, _ in picks["S"]]
        print("    decode: local maxima above 0.3 with >= 50 samples separation")
        print(f"      P picks (sample) = {p_picks[:20]}{' ...' if len(p_picks) > 20 else ''}"
              f"  ({len(p_picks)} total)")
        print(f"      S picks (sample) = {s_picks[:20]}{' ...' if len(s_picks) > 20 else ''}"
              f"  ({len(s_picks)} total)")
    elif task == "S2S_pmp":
        probs = out[0].cpu().numpy()
        print(f"    output: (1, 2) probs = {np.round(probs, 4).tolist()}")
        print(f"    decode: argmax -> class {'up' if probs.argmax() == 0 else 'down'}")
    elif task == "S2S_baz":
        cos, sin = out[0].item(), out[1].item()
        baz = (math.degrees(math.atan2(sin, cos)) + 360.0) % 360.0
        print(f"    output: cos={cos:+.4f}, sin={sin:+.4f}")
        print(f"    decode: baz = atan2(sin, cos) = {baz:.2f} deg")
    elif task == "S2S_dis":
        print(f"    output: distance = {out[0].item():.2f} km")
    elif task == "S2S_bazdis":
        cos, sin, dis = out[0].item(), out[1].item(), out[2].item()
        baz = (math.degrees(math.atan2(sin, cos)) + 360.0) % 360.0
        print(f"    output: cos={cos:+.4f}, sin={sin:+.4f}, dis={dis:.2f} km")
        print(f"    decode: baz = {baz:.2f} deg, dis = {dis:.2f} km")


def main() -> None:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # Test sample (from the test set) in [Z, N, E] order.
    waveform, p_idx, labels, sample_name = load_test_sample(rng)
    window = prepare_p_centered(
        waveform, p_idx=p_idx, in_samples=IN_SAMPLES,
        p_position_ratio=P_POSITION_RATIO, norm_mode="std", normalize_window=False,
    )
    print(f"test sample: {sample_name}")
    if labels:
        pmp = {0: "up", 1: "down"}.get(labels.get("pmp", -1), "unknown")
        print(f"  source={labels['source']} trace={labels['trace_id']} "
              f"P@{p_idx} (label: pmp={pmp}, baz={labels['baz_deg']}, dis={labels['dis_km']})")
    print(f"  preprocessed window: {window.shape}  ([Z, N, E], demeaned + std-normalised, "
          f"P at sample {int(IN_SAMPLES * P_POSITION_RATIO)})")
    print(f"  weights: looking for checkpoints in {os.path.relpath(CHECKPOINT_DIR, REPO_ROOT)}/\n")

    random_tasks = []
    for task, spec in TASK_SPECS.items():
        if task == "S2S_dpk":
            # Phase picking works on any window -> use the full normalised trace,
            # so the picked samples are in trace coordinates.
            x = torch.from_numpy(fit_length(normalize(waveform, "std"), IN_SAMPLES)).unsqueeze(0)
        elif task == "S2S_pmp":
            # Polarity normalises the cut window again (mirrors training).
            x = torch.from_numpy(
                prepare_p_centered(
                    waveform, p_idx=p_idx, in_samples=IN_SAMPLES,
                    p_position_ratio=P_POSITION_RATIO, norm_mode="std",
                    normalize_window=True,
                )
            ).unsqueeze(0)
        else:
            x = torch.from_numpy(window).unsqueeze(0)  # P-centered window (1, 3, 6000)

        model, status = build_for_task(task)
        if "random init" in status:
            random_tasks.append(task)
        model.eval()
        with torch.no_grad():
            out = model(x)

        print(f"[{task}] {spec['description']}")
        print(f"    weights: {status}")
        print(f"    input : {tuple(x.shape)}  [Z, N, E] demeaned + std-normalised")
        describe(task, out)
        print()

    if random_tasks:
        print("=" * 78)
        print("[WARNING] No checkpoint was found for: " + ", ".join(random_tasks))
        print("          These tasks ran with randomly initialised weights, so their")
        print("          predictions are meaningless. Place trained weights at")
        print("          extras/checkpoints/<task>.pth to run with real weights.")
        print("=" * 78)
    else:
        print("All five tasks loaded their trained weights and ran successfully.")


if __name__ == "__main__":
    main()
