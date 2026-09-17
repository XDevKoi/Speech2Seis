# Test sample

`sample_stead_test.npz` is a single random sample drawn from the **STEAD test
split** (`stead_test.csv`). It is the sample used for the single-sample smoke
test.

| File | Source | Split |
|------|--------|-------|
| `sample_stead_test.npz` | STEAD (`B918.PB_20170526084429_EV`) | test |

The waveform is stored in model-ready form: `[Z, N, E]`, 100 Hz, raw amplitude.
Keys:

| Key | Shape / type | Meaning |
|-----|--------------|---------|
| `waveform` | `(3, 6000)` float32 | 3-component waveform in `[Z, N, E]` order, 100 Hz |
| `p_idx` | int | P-arrival sample index |
| `s_idx` | int | S-arrival sample index |
| `sampling_rate` | int | 100 (Hz) |
| `pmp` | int | P-motion polarity label; `-1` = unknown |
| `baz_deg` | float32 | back-azimuth label in degrees |
| `dis_km` | float32 | epicentral distance label in km |
| `source` | str | `stead` |
| `trace_id` | str | STEAD trace identifier |

> STEAD has no first-motion polarity label, so `pmp = -1` here. The polarity task
> (`S2S_pmp`) requires the DiTing dataset, which is not redistributed in this
> repository.

Example:

```python
import numpy as np
import torch
from s2s import build_model, load_checkpoint, prepare_p_centered

sample = np.load("extras/samples/sample_stead_test.npz")
waveform = sample["waveform"]            # (3, 6000) in [Z, N, E]

# P-centered tasks (pmp / baz / dis / bazdis)
window = prepare_p_centered(
    waveform,
    p_idx=int(sample["p_idx"]),
    in_samples=6000,
    p_position_ratio=0.5,
    norm_mode="std",
)
x = torch.from_numpy(window).unsqueeze(0)      # (1, 3, 6000)

model = build_model("S2S_baz")                 # add pretrained_path=... for real weights
model.eval()
with torch.no_grad():
    cos, sin = model(x)
```
