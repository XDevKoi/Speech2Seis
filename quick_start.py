"""Single-sample test for all five Speech2Seis downstream tasks.

The test sample is drawn from the **test set** (see ``extras/samples/``); if
several ``*.npz`` samples are present, one is picked at random.

Input contract (all tasks): a ``(1, 3, 6000)`` float32 tensor with channels in
``[Z, N, E]`` order, demeaned and std-normalised per trace. The P-centered tasks
(``S2S_pmp`` / ``S2S_baz`` / ``S2S_dis`` / ``S2S_bazdis``) additionally require
the P arrival to sit at sample ``0.5 * 6000 = 3000`` (``S2S_dpk`` works on any
window; the P-centered window is used here for all tasks).

By default the models are built with **randomly initialised** weights
(``pretrain=False``): this script only checks the model definitions and the
input/output shapes. See the commented block below to load real weights.

Run:
    python quick_start.py
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

from s2s import TASK_SPECS, build_model, prepare_p_centered  # noqa: E402

IN_SAMPLES = 6000
P_POSITION_RATIO = 0.5
SAMPLE_DIR = os.path.join(REPO_ROOT, "extras", "samples")

# ---------------------------------------------------------------------------
# To run with real weights, uncomment the block below and fill in the paths.
#
#   BACKBONE_PATH = ""                 # local Wav2Vec2 dir, or set S2S_WAV2VEC2_PATH
#   CHECKPOINT = {                     # see extras/checkpoints/README.md
#       "S2S_dpk":    "",
#       "S2S_pmp":    "",
#       "S2S_baz":    "",
#       "S2S_dis":    "",
#       "S2S_bazdis": "",
#   }
#   model = build_model(task, pretrained_path=BACKBONE_PATH)   # pretrain=True
#   load_checkpoint(model, CHECKPOINT[task])
# ---------------------------------------------------------------------------
PRETRAINED_WEIGHTS_LOADED = False


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


def describe(task: str, out) -> None:
    """Print the decoded output of ``task`` for a single sample."""
    if task == "S2S_dpk":
        probs = out[0].cpu().numpy()  # (3, L): [N, P, S]
        print(f"    output: (1, 3, {probs.shape[-1]}) probs -> "
              f"mean[N,P,S] = [{probs[0].mean():.3f}, {probs[1].mean():.3f}, {probs[2].mean():.3f}]")
        print("    decode: pick peaks of the P/S curves above 0.3")
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

    if not PRETRAINED_WEIGHTS_LOADED:
        print("=" * 78)
        print("[WARNING] No pretrained backbone / task checkpoint is loaded.")
        print("          Models use randomly initialised weights (`pretrain=False`),")
        print("          so the predictions below are meaningless -- this is only a")
        print("          shape / definition check. Uncomment the weight-loading")
        print("          block in quick_start.py to run with real weights.")
        print("=" * 78 + "\n")

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
          f"P at sample {int(IN_SAMPLES * P_POSITION_RATIO)})\n")

    for task, spec in TASK_SPECS.items():
        # S2S_pmp normalises the cut window again (mirrors training); others don't.
        if task == "S2S_pmp":
            x = torch.from_numpy(
                prepare_p_centered(
                    waveform, p_idx=p_idx, in_samples=IN_SAMPLES,
                    p_position_ratio=P_POSITION_RATIO, norm_mode="std",
                    normalize_window=True,
                )
            ).unsqueeze(0)
        else:
            x = torch.from_numpy(window).unsqueeze(0)  # (1, 3, 6000)

        model = build_model(task, pretrain=False, freeze=False)
        model.eval()
        with torch.no_grad():
            out = model(x)

        print(f"[{task}] {spec['description']}")
        print(f"    input : {tuple(x.shape)}  [Z, N, E] demeaned + std-normalised")
        describe(task, out)
        print()

    print("All five tasks ran successfully. See README.md for the I/O contract.")


if __name__ == "__main__":
    main()
