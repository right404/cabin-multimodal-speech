# 视觉先验引导的宽带 SBL-INCM-MVDR

## 方法

本模块参考论文 *Sparse Bayesian learning for interference-plus-noise covariance matrix
reconstruction in robust MVDR beamforming*，实现：

- 式（24）–（28）：MMV-SBL 后验和角功率谱；
- 式（31）：目标扇区内主导方向估计；
- 式（34）：噪声方差校准的峰值筛选；
- 式（39）–（41）：能量覆盖与方向—尺度解耦的干扰协方差重构；
- 式（42）–（44）：INCM、对角加载和 MVDR 权重；
- 附录 A：残差型噪声方差更新。

论文方法面向窄带阵列快拍。本项目通过 STFT 将其扩展到宽带语音：每个频点独立估计
SBL 空间谱和 MVDR 权重，再通过 iSTFT 重建时域语音。

## 视觉先验

视觉模块先确定目标人物，再通过会议座位配置得到目标方位。该方向在阵列处理中有两个用途：

1. 计算目标导向矢量；
2. 建立目标保护扇区，避免将目标语音写入干扰协方差。

`adaptive_sector_half_width` 支持根据方向置信度调整扇区宽度：置信度高时缩小搜索范围，
置信度降低时扩大目标保护区域。这是我在宽带语音场景中加入的扩展。

## 代码映射

| 方法步骤 | 代码 |
|---|---|
| MMV-SBL | `run_mmv_sbl` |
| 干扰筛选与 INCM 重构 | `reconstruct_incm_mvdr` |
| 置信度自适应保护扇区 | `adaptive_sector_half_width` |
| STFT 宽带化与 iSTFT 重建 | `wideband_sbl_mvdr` |
| 多声源仿真评测 | `scripts/run_sbl_mvdr_demo.py` |

`frequency_stride` 用于控制宽带计算量。选定频点运行完整 SBL-INCM-MVDR，其余频点使用
同一视觉方向的频域 Delay-and-Sum，最终共同参与 iSTFT 重建。

## 快速验证

```powershell
conda activate cabin_avspeech
python scripts/run_sbl_mvdr_demo.py
```

输出目录：

```text
artifacts/sbl_mvdr_demo/
├── reference.wav
├── mixture.wav
├── delay_sum.wav
├── sbl_incm_mvdr.wav
└── metrics.json
```

使用 1° 角度网格和 200 次最大迭代：

```powershell
python scripts/run_sbl_mvdr_demo.py --paper-settings
```

## 真实数据入口

- AMI：视觉人物身份驱动目标保护扇区，结果见
  [`REAL_MULTIMODAL_AMI.md`](REAL_MULTIMODAL_AMI.md)；
- AISHELL-4：SRP-PHAT 方向置信度驱动自适应扇区和后端门控，结果见
  [`REAL_DATA_AISHELL4.md`](REAL_DATA_AISHELL4.md)。

两个入口共用 `src/cabin_speech/sbl_mvdr.py`，分别验证多模态目标选择和真实中文会议阵列增强。
