# extras

Non-source files for Speech2Seis. 中文说明见 [README.zh-CN.md](README.zh-CN.md)。

```
extras/
├── README.zh-CN.md  # Chinese documentation
├── samples/         # STEAD test-set sample for running the models
├── seisbench/       # SeisBench extension for the S2S_dpk phase picker
└── checkpoints/     # released weights (placeholder; see its README)
```

| Item | Description |
|------|-------------|
| `samples/` | One random sample from the STEAD test split, with label metadata. |
| `seisbench/s2s_dpk.py` | `WaveformModel` extension exposing `S2S_dpk` through the SeisBench API (`annotate` / `classify`). |
| `checkpoints/` | Empty for now; download links for the backbone and task weights go in `checkpoints/README.md`. |
