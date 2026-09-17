# Speech2Seis (S2S)

[English](../README.md) | 中文文档

**作者：** 夏登科<sup>1</sup>，房立华<sup>2</sup>，刘烽<sup>3</sup>

<sup>1</sup> 中国地震局地球物理研究所<br>
<sup>2</sup> 中国地震局地震预测研究所<br>
<sup>3</sup> 上海人工智能实验室

> 论文正在撰写中，待投。

**Speech2Seis** 探索地震监测中的**跨模态迁移**：将语音领域预训练模型
（**Wav2Vec 2.0**）通过轻量级多尺度卷积嵌入器与参数高效微调（LoRA）迁移到
三分量地震波形上。同一套共享骨干网络支撑五个下游任务：

| 任务 | 模型 |
|------|------|
| 震相拾取（P / S） | `S2S_dpk` |
| P 波初动极性分类 | `S2S_pmp` |
| 反方位角估计 | `S2S_baz` |
| 震中距估计 | `S2S_dis` |
| 反方位角 + 震中距联合估计 | `S2S_bazdis` |

模型 state_dict 结构与参考训练实现一致，训练好的权重可直接加载，无需键名重映射。

---

## 模型架构

```
waveform (B, 3, 6000)  [Z, N, E]
        │
        ▼
多尺度卷积嵌入器                            # 每级并行卷积核 {16,24,32,40}
        │  conv_strides = [2, 2, 1, 1]      # 总下采样 ×4
        ▼
(B, 96, 1500)
        │  patch 切分 (patch_size = 8)
        ▼
(B, 187, 768)  ──►  预训练 Wav2Vec2 Transformer Encoder  ──►  (B, 187, 768)
                    （绕过其 CNN 前端，采用 LoRA 微调）
        │
        ▼
( B, 96, 1500 )  ──►  任务头
```

* **多尺度卷积嵌入器** —— 每级并联 4 个不同卷积核的 `Conv1d` 分支，并用 1×1
  卷积融合。最后一级使用 `stride=1` 以保留时序分辨率，适配 Wav2Vec2 的位置
  编码（卷积核大小 128）。
* **预训练编码器** —— 仅使用 Wav2Vec2 的 Transformer Encoder，丢弃其卷积特征
  提取器；注入 LoRA（`r=16, alpha=16, dropout=0.1, target_modules="all-linear"`），
  仅训练 LoRA 与 LayerNorm 参数。
* **任务头** —— 震相拾取使用密集上采样解码头；窗口级任务（极性、方位角、
  震中距）使用 **P 波居中局部头**：以 P 到时为中心裁剪固定长度窗口
  （`p_position_ratio=0.5`，`local_half_width=128`），避免目标信号被整条波形平均掉。

---

## 仓库结构

```
Speech2Seis/
├── quick_start.py           # 五个任务的单样本测试
├── s2s/
│   ├── __init__.py          # 对外 API
│   ├── backbone.py          # Speech2Seis 骨干（卷积嵌入 + Wav2Vec2 编码器）
│   ├── heads.py             # 任务头（拾取 / 局部 pmp、baz、dis、bazdis）
│   ├── models.py            # S2S_dpk / S2S_pmp / S2S_baz / S2S_dis / S2S_bazdis
│   ├── losses.py            # BCELoss / CELoss / BAZDisLoss
│   └── preprocess.py        # 归一化 + P 波居中切窗
├── extras/                  # 源码之外的文件
│   ├── samples/             # STEAD 测试集样本
│   ├── seisbench/           # S2S_dpk 的 SeisBench 扩展
│   └── checkpoints/         # 已发布权重（下载链接）
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## 安装

```bash
pip install -r requirements.txt
# 或以可编辑模式安装
pip install -e .
```

预训练语音骨干（Wav2Vec2）下载地址：

> **下载：** https://huggingface.co/facebook/wav2vec2-base-960h/tree/main

路径解析顺序：

1. `pretrained_path` 参数；
2. 环境变量 `S2S_WAV2VEC2_PATH` / `WAV2VEC2_PATH`；
3. HuggingFace hub id `facebook/wav2vec2-base-960h`。

```bash
# 示例：使用本地副本（便于离线）
export S2S_WAV2VEC2_PATH=/path/to/wav2vec2
```

### 硬件要求

> **Speech2Seis 属于亿级大模型（约 90M 参数），强烈建议使用 GPU 加速推理。**
> 显存 **≥ 8 GB** 即可支持单次批大小为 32。

---

## 模型与输入/输出约定

**所有输入**均为 `float32` 张量，形状 `(B, 3, 6000)`，通道顺序为 **`[Z, N, E]`**，
并已按通道**去均值**与**标准差归一化**。`L` 表示波形长度（6000）。P 波居中类
任务要求 P 到时位于 `0.5 * in_samples`（即样点 3000）处，见
`s2s.preprocess.prepare_p_centered`。

| 模型 | 输入 | 输出 | 损失 | 解码 |
|------|------|------|------|------|
| `S2S_dpk` | `(B, 3, 6000)` | `(B, 3, L)` sigmoid，`[N, P, S]` | 加权 BCE `[0, 1, 1]` | 取 P/S 曲线中 >0.3 的局部极大值（相邻间隔 ≥50 样点），输出到时样点索引（`s2s.postprocess.decode_dpk`） |
| `S2S_pmp` | `(B, 3, 6000)`，P 居中 | `(B, 2)` softmax，`[up, down]` | 交叉熵 `[1, 1]` | `argmax`（0 = up，1 = down） |
| `S2S_baz` | `(B, 3, 6000)`，P 居中 | `(cos, sin)`，各 `(B, 1)` | 对 `(cos, sin)` 的 Huber | `atan2(sin, cos)·180/π mod 360` |
| `S2S_dis` | `(B, 3, 6000)`，P 居中 | `(B, 1)` km | Huber | 直接输出（sigmoid × 500） |
| `S2S_bazdis` | `(B, 3, 6000)`，P 居中 | `(cos, sin, dis)` | `BAZDisLoss(w_baz=0.95, w_dis=0.05)` | baz 用 `atan2`，dis 为 km |

`S2S_baz` 返回**二元组** `(cos, sin)`；`S2S_bazdis` 返回**三元组**
`(cos, sin, dis)`。也可通过代码获取相同约定：

```python
from s2s import TASK_SPECS
print(TASK_SPECS["S2S_bazdis"])
```

### 训练目标（已发布权重所用）

```python
import torch.nn as nn
from s2s.losses import BCELoss, CELoss, BAZDisLoss

# S2S_dpk:    BCELoss(weight=[[0], [1], [1]])                 # [N, P, S]
# S2S_pmp:    CELoss(weight=[1, 1])                           # [up, down]
# S2S_baz:    nn.HuberLoss()(cos) + nn.HuberLoss()(sin)       # 目标为 (cos, sin)
# S2S_dis:    nn.HuberLoss()                                  # 震中距 km
# S2S_bazdis: BAZDisLoss(w_baz=0.95, w_dis=0.05)              # (cos, sin, dis)
```

---

## 使用方法

### 1. 快速上手（单样本测试）

将单个 `(1, 3, 6000)` 样本依次通过五个任务：

```bash
python quick_start.py
```

测试样本来自 **STEAD 测试集**（见 `extras/samples/`，为随机抽取的一条）。
`quick_start.py` 默认会在 `extras/checkpoints/` 中查找训练好的权重：

* 若存在 `extras/checkpoints/<task>.pth`（如 `S2S_dpk.pth`），则加载该权重 ——
  预训练 Wav2Vec2 骨干通过 `S2S_WAV2VEC2_PATH` / `pretrained_path` 解析，
  否则从 HuggingFace 下载；
* 若某任务没有对应权重，则使用**随机初始化**权重，并打印提示（仅用于检查
  形状与模型定义）。

预训练任务权重可从
[**Google Drive**](https://drive.google.com/drive/folders/1ZdSloEIN6NLm9lvs7r9RAs26K3_p4bot?usp=sharing)
下载，放入 `extras/checkpoints/` 并命名为 `<task>.pth`。

因此，只要把权重放进 `extras/checkpoints/`，直接运行 `python quick_start.py`
即可用真实权重在测试集样本上测试。其中 `S2S_dpk` 会用
`s2s.postprocess.decode_dpk` 解码为 P/S 到时样点索引（>0.3 的局部极大值，
相邻间隔 ≥50 样点）并打印。等价的显式代码如下：

```python
import torch
from s2s import build_model, load_checkpoint

model = build_model("S2S_bazdis", pretrained_path="/path/to/wav2vec2")
load_checkpoint(model, "extras/checkpoints/S2S_bazdis.pth")
model.eval()

x = torch.randn(1, 3, 6000)        # (B, 3, 6000)，[Z, N, E]，去均值 + 标准差归一化
with torch.no_grad():
    cos, sin, dis = model(x)
```

### 2. 批量运行建议

**`S2S_dpk` —— 使用 SeisBench。** `extras/seisbench/s2s_dpk.py` 提供
`seisbench.models.base.WaveformModel`，SeisBench 的 `annotate` / `classify`
已自动处理长波形、滑窗、重叠、blinding 与批处理：

```python
import sys
sys.path.insert(0, "extras/seisbench")
from s2s_dpk import S2S_dpk

model = S2S_dpk(pretrained_path="/path/to/wav2vec2")
# load_checkpoint(model, "extras/checkpoints/S2S_dpk.pth")
model.eval()

annotations = model.annotate(stream)                                # 概率曲线
output = model.classify(stream, P_threshold=0.3, S_threshold=0.3)   # 拾取结果
for pick in output.picks:
    print(pick.phase, pick.peak_time, pick.peak_value)
```

若需集成进 `seisbench.models`：

1. 将 `extras/seisbench/s2s_dpk.py` 复制到 `seisbench/models/`（需保证 `s2s`
   可被 import）；
2. 在 `seisbench/models/__init__.py` 中加入 `from .s2s_dpk import S2S_dpk`。

**`S2S_pmp` / `S2S_baz` / `S2S_dis` / `S2S_bazdis` —— 沿用现有代码即可。**
这些 P 波居中任务需要逐样本的 P 居中切窗与各自的解码方式，因此保留现有的
数据集 / 切窗 / 批量推理代码，仅通过 `build_model` + `load_checkpoint` 替换为
新模型；随后按现有代码的方式将多个窗口堆叠为 `(B, 3, 6000)` 即可。

```python
import numpy as np
import torch
from s2s import build_model, prepare_p_centered

# waveform: (3, L)，顺序 [Z, N, E]；p_idx: P 波到样点索引
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

## 训练（参考）

### 训练数据

| 任务 | 训练集 | 验证集 | 数据来源 |
|------|--------|--------|----------|
| `S2S_dpk`、`S2S_baz`、`S2S_dis`、`S2S_bazdis` | 1,000,000 | 20,000 | STEAD + DiTing 1.0，各占一半（500k / 500k） |
| `S2S_pmp` | 300,000 | 10,000 | 全部来自 DiTing 1.0 |

### 超参数

已发布权重的训练配置如下：

| 设置 | 取值 |
|------|------|
| 输入 | 6000 样点 @ 100 Hz，`[Z, N, E]`，去均值 + 标准差归一化 |
| P 波居中窗口 | `p_position_ratio = 0.5`，`local_half_width = 128` |
| Batch size | 128 × 4 GPU（DDP） |
| 优化器 | Adam，weight decay 0 |
| 学习率 | CyclicLR，base 1e-4–5e-4 → max 5e-4–1e-3，warmup 4500 / down 5000 |
| Epochs | 200（早停 patience 30） |
| LoRA | `r=16, alpha=16, dropout=0.1, target_modules="all-linear"` |

---

## Acknowledgement

**特别感谢 [SeisMoLLM](https://github.com/StarMoonWang/SeisMoLLM)**：该工作在地震监测
**跨模态迁移方面做出了特别贡献**，为本工作（Speech2Seis）奠定了基础。

本工作同时基于：

* [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) —— 预训练语音模型。
* [SeisBench](https://github.com/seisbench/seisbench) —— `extras/seisbench/s2s_dpk.py`
  使用的模型接口。

## 许可证

MIT，详见 [LICENSE](../LICENSE)。
