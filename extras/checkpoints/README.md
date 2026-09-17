# Checkpoints

Download links for the pretrained backbone and the released task weights go
here. Local `*.pth` files are git-ignored (they are large), so weights are
distributed through the links below rather than committed to the repository.

## Backbone (Wav2Vec2)

| Item | URL |
|------|-----|
| Pretrained Speech2Seis backbone (Wav2Vec2) | https://huggingface.co/facebook/wav2vec2-base-960h/tree/main |

## Task weights

| Model | URL |
|-------|-----|
| `S2S_dpk` | <!-- TODO: fill in download URL --> |
| `S2S_pmp` | <!-- TODO: fill in download URL --> |
| `S2S_baz` | <!-- TODO: fill in download URL --> |
| `S2S_dis` | <!-- TODO: fill in download URL --> |
| `S2S_bazdis` | <!-- TODO: fill in download URL --> |

## Loading

```python
import torch
from s2s import build_model, load_checkpoint

model = build_model("S2S_dpk", pretrained_path="/path/to/wav2vec2")  # or set S2S_WAV2VEC2_PATH
load_checkpoint(model, "extras/checkpoints/S2S_dpk.pth")
model.eval()
```

Each checkpoint may be a raw `state_dict` or wrap it as `{"model_dict": ...}`;
`load_checkpoint` handles both and strips `module.` / `_orig_mod.` prefixes.
