# AISHELL-4 真实八通道阵列实验

## 实验目的

AMI 实验验证视觉身份驱动的目标选择；AISHELL-4 实验用于补充验证阵列增强后端在另一套
真实中文会议圆阵数据上的表现。

数据来自 [OpenSLR SLR111](https://www.openslr.org/111/)，许可为 CC BY-SA 4.0。test
压缩包约 5.2 GB，包含：

- 20 场真实中文会议；
- 16 kHz、8 通道圆形麦克风阵列 FLAC；
- 逐说话人的 Praat TextGrid；
- RTTM 说话人时间标注。

原始数据位于 `data/raw/aishell4/test/`，按数据集许可单独下载。

## 接入内容

- 通用二维阵列几何和八通道均匀圆阵；
- 支持时间寻址的多通道 WAV/FLAC 加载器；
- TextGrid 解析和无重叠语音片段筛选；
- SRP-PHAT 宽带方向先验；
- 圆阵宽带 SBL-INCM-MVDR；
- Whisper-small 中文 ASR 和繁简统一 CER；
- 固定扇区、自适应扇区与置信度门控对照。

AISHELL-4 提供多通道音频和说话标注。这里使用 SRP-PHAT 方向及置信度驱动同一个
SBL-INCM 后端，用于检验保护扇区和置信度门控策略。

## 评测设置

每场会议自动选择第一个时长 3–7 秒、具有有效文本且无其他说话人重叠的片段：

- 5 场开发子集用于确定置信度阈值 `0.20`；
- 其余 15 场作为留出子集；
- Whisper 输出去除标注标签和标点，并统一为简体中文后计算 CER。

## 留出 15 场结果

| 方法 | 平均 CER |
|---|---:|
| 单通道 channel 0 | 43.67% |
| Delay-and-Sum | 38.88% |
| SBL-INCM，固定 5° 目标扇区 | 41.71% |
| SBL-INCM，置信度自适应扇区 | 38.94% |
| 置信度门控：DS / 自适应 SBL | **38.78%** |

置信度门控方法相对单通道的 CER 降低 **11.20%**。结果表明，方向置信度可以同时用于
调整目标保护区和选择阵列后端：高置信度时使用自适应 SBL-INCM，低置信度时采用
Delay-and-Sum。

完整逐会议转写和指标：

```text
artifacts/aishell4_batch/heldout_15_metrics.json
```

## 复现

检查语料：

```powershell
python scripts/inspect_aishell4.py --root data/raw/aishell4/test
```

运行单片段：

```powershell
python scripts/evaluate_aishell4_segment.py --enable-asr
```

运行开发集与留出集：

```powershell
python scripts/evaluate_aishell4_batch.py --segments 5

python scripts/evaluate_aishell4_batch.py --segments 15 --session-offset 5 `
  --output artifacts/aishell4_batch/heldout_15_metrics.json
```

重新计算文本标准化后的 CER：

```powershell
python scripts/recompute_cer_report.py `
  artifacts/aishell4_batch/heldout_15_metrics.json
```

该实验与 AMI 共用阵列几何、SRP-PHAT、Delay-and-Sum、SBL-INCM-MVDR 和评测基础模块，
展示了后端在不同真实八通道圆阵语料上的复用方式。
