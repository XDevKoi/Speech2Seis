# Checkpoints

Download links for the pretrained backbone and the released task weights go
here. `quick_start.py` automatically looks for `<task>.pth` (e.g. `S2S_dpk.pth`)
in this folder and loads it when present. Local `*.pth` files are git-ignored
(they are large), so weights are distributed through the links below rather than
committed to the repository.

## Backbone (Wav2Vec2)

| Item | URL |
|------|-----|
| Pretrained Speech2Seis backbone (Wav2Vec2) | https://huggingface.co/facebook/wav2vec2-base-960h/tree/main |

## Task weights

All released task weights (`S2S_dpk`, `S2S_pmp`, `S2S_baz`, `S2S_dis`,
`S2S_bazdis`) are available at:

**https://drive.google.com/drive/folders/1ZdSloEIN6NLm9lvs7r9RAs26K3_p4bot?usp=sharing**

| Model | Download |
|-------|----------|
| `S2S_dpk` | [Google Drive](https://drive.google.com/drive/folders/1ZdSloEIN6NLm9lvs7r9RAs26K3_p4bot?usp=sharing) |
| `S2S_pmp` | [Google Drive](https://drive.google.com/drive/folders/1ZdSloEIN6NLm9lvs7r9RAs26K3_p4bot?usp=sharing) |
| `S2S_baz` | [Google Drive](https://drive.google.com/drive/folders/1ZdSloEIN6NLm9lvs7r9RAs26K3_p4bot?usp=sharing) |
| `S2S_dis` | [Google Drive](https://drive.google.com/drive/folders/1ZdSloEIN6NLm9lvs7r9RAs26K3_p4bot?usp=sharing) |
| `S2S_bazdis` | [Google Drive](https://drive.google.com/drive/folders/1ZdSloEIN6NLm9lvs7r9RAs26K3_p4bot?usp=sharing) |

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
