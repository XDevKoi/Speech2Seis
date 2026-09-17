# Speech2Seis (S2S)

English | [中文文档](extras/README.zh-CN.md)

**Authors:** Dengke Xia<sup>1</sup>, Lihua Fang<sup>2</sup>, Feng Liu<sup>3</sup>

<sup>1</sup> Institute of Geophysics, China Earthquake Administration<br>
<sup>2</sup> Institute of Earthquake Forecasting, China Earthquake Administration<br>
<sup>3</sup> Shanghai AI Laboratory

> A paper is in preparation and will be submitted.

**Speech2Seis** explores **cross-modal transfer** for seismic monitoring: a
pretrained speech model (**Wav2Vec2 2.0**) is adapted to 3-component seismic
waveforms through a lightweight multi-scale convolutional embedder and
parameter-efficient fine-tuning (LoRA). A single shared backbone handles five
downstream tasks:

| Task | Model |
|------|-------|
| Phase picking (P / S) | `S2S_dpk` |
| P-motion polarity classification | `S2S_pmp` |
| Back-azimuth estimation | `S2S_baz` |
| Epicentral distance estimation | `S2S_dis` |
| Joint back-azimuth + distance | `S2S_bazdis` |

The state-dict layout matches the reference training implementation, so trained
checkpoints load without key remapping.

---

## Architecture

```
waveform (B, 3, 6000)  [Z, N, E]
        │
        ▼
Multi-Scale Convolutional Embedder        # parallel kernels {16,24,32,40} per stage
        │  conv_strides = [2, 2, 1, 1]     # total downsample ×4
        ▼
(B, 96, 1500)
        │  patchify (patch_size = 8)
        ▼
(B, 187, 768)  ──►  Pretrained Wav2Vec2 Transformer encoder  ──►  (B, 187, 768)
                    (CNN front-end bypassed, LoRA fine-tuned)
        │
        ▼
( B, 96, 1500 )  ──►  task head
```

* **Multi-scale conv embedder** – each stage runs 4 parallel `Conv1d` branches
  with increasing kernel sizes and fuses them with a 1×1 convolution. The last
  stage uses `stride=1` to preserve temporal resolution for the Wav2Vec2
  positional encoding (kernel size 128).
* **Pretrained encoder** – only the Transformer encoder of Wav2Vec2 is used;
  its convolutional feature extractor is discarded. LoRA
  (`r=16, alpha=16, dropout=0.1, target_modules="all-linear"`) is applied and
  only LoRA + LayerNorm parameters are trained.
* **Task heads** – a dense up-sampling decoder for phase picking, and
  **P-centered local** heads for the window-level tasks (polarity, azimuth,
  distance). The local heads crop a fixed window around the P arrival
  (`p_position_ratio=0.5`, `local_half_width=128`) so the target signal is not
  averaged out over the whole trace.

---

## Repository structure

```
Speech2Seis/
├── quick_start.py           # single-sample test for all five tasks
├── s2s/
│   ├── __init__.py          # public API
│   ├── backbone.py          # Speech2Seis backbone (conv embedder + Wav2Vec2 encoder)
│   ├── heads.py             # task heads (detection/picking, local pmp/baz/dis/bazdis)
│   ├── models.py            # S2S_dpk / S2S_pmp / S2S_baz / S2S_dis / S2S_bazdis
│   ├── losses.py            # BCELoss / CELoss / BAZDisLoss
│   └── preprocess.py        # normalize + P-centered windowing
├── extras/                  # non-source files
│   ├── samples/             # STEAD test-set sample
│   ├── seisbench/           # S2S_dpk SeisBench extension
│   └── checkpoints/         # released weights (placeholder)
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Installation

```bash
pip install -r requirements.txt
# or, to install the package (editable)
pip install -e .
```

The pretrained speech backbone (Wav2Vec2) can be downloaded from:

> **Download:** https://huggingface.co/facebook/wav2vec2-base-960h/tree/main

The path is resolved in this order:

1. the `pretrained_path` argument,
2. the `S2S_WAV2VEC2_PATH` / `WAV2VEC2_PATH` environment variable,
3. the HuggingFace hub id `facebook/wav2vec2-base-960h`.

```bash
# example: use a local copy (offline friendly)
export S2S_WAV2VEC2_PATH=/path/to/wav2vec2
```

---

## Model zoo and I/O contract

**All inputs** are `float32` tensors of shape `(B, 3, 6000)` with channels in
**`[Z, N, E]` order**, **demeaned** and **std-normalised** per trace. `L` denotes
the waveform length (6000). The P-centered models require the P arrival to sit at
`0.5 * in_samples` (sample 3000) — see `s2s.preprocess.prepare_p_centered`.

| Model | Input | Output | Loss | Decoding |
|-------|-------|--------|------|----------|
| `S2S_dpk` | `(B, 3, 6000)` | `(B, 3, L)` sigmoid, `[N, P, S]` | weighted BCE `[0, 1, 1]` | peak-pick P/S curves above 0.3 |
| `S2S_pmp` | `(B, 3, 6000)`, P-centered | `(B, 2)` softmax, `[up, down]` | cross entropy `[1, 1]` | `argmax` (0 = up, 1 = down) |
| `S2S_baz` | `(B, 3, 6000)`, P-centered | `(cos, sin)`, each `(B, 1)` | Huber on `(cos, sin)` | `atan2(sin, cos)·180/π mod 360` |
| `S2S_dis` | `(B, 3, 6000)`, P-centered | `(B, 1)` km | Huber | direct (sigmoid × 500) |
| `S2S_bazdis` | `(B, 3, 6000)`, P-centered | `(cos, sin, dis)` | `BAZDisLoss(w_baz=0.95, w_dis=0.05)` | `atan2` for baz + km for dis |

`S2S_baz` returns a **tuple** `(cos, sin)`; `S2S_bazdis` returns a **3-tuple**
`(cos, sin, dis)`. The same contract is available programmatically:

```python
from s2s import TASK_SPECS
print(TASK_SPECS["S2S_bazdis"])
```

### Training objectives (used for the released checkpoints)

```python
import torch.nn as nn
from s2s.losses import BCELoss, CELoss, BAZDisLoss

# S2S_dpk:    BCELoss(weight=[[0], [1], [1]])                 # [N, P, S]
# S2S_pmp:    CELoss(weight=[1, 1])                           # [up, down]
# S2S_baz:    nn.HuberLoss()(cos) + nn.HuberLoss()(sin)       # targets = (cos, sin)
# S2S_dis:    nn.HuberLoss()                                  # distance in km
# S2S_bazdis: BAZDisLoss(w_baz=0.95, w_dis=0.05)              # (cos, sin, dis)
```

---

## Usage

### 1. Quick start (single-sample test)

Run one `(1, 3, 6000)` sample through all five tasks:

```bash
python quick_start.py
```

The sample is drawn from the **STEAD test set** (see `extras/samples/`).
`quick_start.py` builds the models with randomly initialised weights by default
and prints a warning that no pretrained weights are loaded; it only checks the
definitions and I/O shapes. To run a **single sample with real weights**:

```python
import torch
from s2s import build_model, load_checkpoint

model = build_model("S2S_bazdis", pretrained_path="/path/to/wav2vec2")
load_checkpoint(model, "extras/checkpoints/S2S_bazdis.pth")
model.eval()

x = torch.randn(1, 3, 6000)        # (B, 3, 6000), [Z, N, E], demeaned + std-normalised
with torch.no_grad():
    cos, sin, dis = model(x)
```

### 2. Batch processing recommendations

**`S2S_dpk` — use SeisBench.** `extras/seisbench/s2s_dpk.py` exposes a
`seisbench.models.base.WaveformModel`, and SeisBench's `annotate` / `classify`
already handle long streams, sliding windows, overlap, blinding and batching:

```python
import sys
sys.path.insert(0, "extras/seisbench")
from s2s_dpk import S2S_dpk

model = S2S_dpk(pretrained_path="/path/to/wav2vec2")
# load_checkpoint(model, "extras/checkpoints/S2S_dpk.pth")
model.eval()

annotations = model.annotate(stream)                                # probability curves
output = model.classify(stream, P_threshold=0.3, S_threshold=0.3)   # picks
for pick in output.picks:
    print(pick.phase, pick.peak_time, pick.peak_value)
```

To register it inside `seisbench.models`:

1. copy `extras/seisbench/s2s_dpk.py` into `seisbench/models/` (the `s2s`
   package must be importable),
2. add `from .s2s_dpk import S2S_dpk` to `seisbench/models/__init__.py`.

**`S2S_pmp` / `S2S_baz` / `S2S_dis` / `S2S_bazdis` — reuse the existing code.**
These P-centered tasks need per-sample P-centered windows and task-specific
decoding, so keep the existing dataset / windowing / batched-inference code and
only swap in the new model through `build_model` + `load_checkpoint`; then stack
multiple windows into `(B, 3, 6000)` exactly as the existing code already does.

```python
import numpy as np
import torch
from s2s import build_model, prepare_p_centered

# waveform: (3, L) in [Z, N, E] order; p_idx: P-arrival sample index
window = prepare_p_centered(waveform, p_idx=p_idx, in_samples=6000,
                            p_position_ratio=0.5, norm_mode="std",
                            normalize_window=(task == "S2S_pmp"))
x = torch.from_numpy(window).unsqueeze(0)          # (1, 3, 6000)

model = build_model("S2S_pmp", pretrained_path="/path/to/wav2vec2")
model.eval()
with torch.no_grad():
    out = model(x)
```

---

## Training (reference)

### Data

| Task | Train | Validation | Source |
|------|-------|------------|--------|
| `S2S_dpk`, `S2S_baz`, `S2S_dis`, `S2S_bazdis` | 1,000,000 | 20,000 | STEAD + DiTing 1.0, half each (500k / 500k) |
| `S2S_pmp` | 300,000 | 10,000 | DiTing 1.0 |

### Hyperparameters

The released checkpoints were trained with:

| Setting | Value |
|---------|-------|
| Input | 6000 samples @ 100 Hz, `[Z, N, E]`, demeaned + std-normalised |
| P-centered window | `p_position_ratio = 0.5`, `local_half_width = 128` |
| Batch size | 128 × 4 GPUs (DDP) |
| Optimizer | Adam, weight decay 0 |
| LR schedule | CyclicLR, base 1e-4–5e-4 → max 5e-4–1e-3, warmup 4500 / down 5000 |
| Epochs | 200 (early stop patience 30) |
| LoRA | `r=16, alpha=16, dropout=0.1, target_modules="all-linear"` |

---

## Acknowledgement

We owe **special thanks to
[SeisMoLLM](https://github.com/StarMoonWang/SeisMoLLM)** for its **special
contribution to cross-modal transfer for seismic monitoring**, which laid the
foundation for Speech2Seis.

This work also builds on:

* [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) — the pretrained speech model.
* [SeisBench](https://github.com/seisbench/seisbench) — the model interface used
  by `extras/seisbench/s2s_dpk.py`.

## License

MIT — see [LICENSE](LICENSE).
