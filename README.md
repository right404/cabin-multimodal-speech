# 视听多模态目标说话人增强

在多人会议里，声源定位能告诉我“哪个方向最响”，却不一定能告诉我“我想听谁”。
如果旁边的人声音更大，纯音频定位很容易把波束指向干扰者。

我做这个项目，是想把这两个问题分开：

1. 先通过视频判断正在说话的人是谁；
2. 再把这个人的方向交给麦克风阵列，完成定位和波束形成。

项目使用 AMI Meeting Corpus 的真实会议数据，打通了 4 路人物视频、8 通道圆形阵列、
4 路头戴麦克风和人工标注。视觉侧使用 YuNet 与 Light-ASD，阵列侧实现了
SRP-PHAT、Delay-and-Sum，以及我从论文方法迁移到宽带语音的 SBL-INCM-MVDR。

## 实验结果

### 主动说话人识别

我从 AMI `ES2002a` 的人工标注中自动选取了 16 个无重叠语音片段：

| 数据范围 | 结果 |
|---|---:|
| 全部 16 个片段 | 9 / 16 |
| 目标人物出现在画面中的 10 个片段 | **9 / 10** |
| 有效视觉覆盖率 | 62.5% |

这次实验也暴露了一个很实际的问题：近景摄像头有时会转向白板，人物不在画面里时，
主动说话人模型就没有足够的视觉信息。逐片段结果保存在
[`active_speaker_results.json`](artifacts/ami/active_speaker_results.json)。

### +6 dB 强方向干扰

为了验证视觉信息是否真的改变了目标选择，我把两个真实的 AMI 八通道房间录音做了
可控叠加：目标人物 B 位于 `14°`，干扰人物 D 位于 `−172°`，并让干扰比目标强 `6 dB`。

纯音频 SRP-PHAT 的峰值落在了干扰者的 `−172°`；Light-ASD 仍然选中了目标人物 B，
因此视觉引导链路把波束保持在 `14°`。

| 前端 | 对齐 SI-SDR | 相对纯音频 DOA + DS |
|---|---:|---:|
| 单通道 | -12.37 dB | -0.06 dB |
| 纯音频 DOA + Delay-and-Sum | -12.31 dB | 基准 |
| 视觉目标 + Delay-and-Sum | -10.11 dB | **+2.20 dB** |
| 视觉目标 + SBL-INCM-MVDR | -10.17 dB | **+2.15 dB** |

我更看重这组结果里“目标选对了”这件事：两种视觉引导后端得到接近的增益，说明这次
提升主要来自视觉提供的人物身份，而不是单纯更换波束形成器。完整参数与中间结果见
[`interference_results.json`](artifacts/ami/interference_results.json)。

## 我实现的处理链路

```mermaid
flowchart LR
    V[4 路人物视频] --> Y[YuNet 人脸序列]
    A1[阵列通道 1] --> ASD[Light-ASD]
    Y --> ASD
    ASD --> ID[目标人物]
    ID --> ANGLE[人物到方位角标定]

    A8[8 通道圆形阵列] --> SRP[SRP-PHAT]
    A8 --> DS[Delay-and-Sum]
    A8 --> SBL[MMV-SBL 空间谱]
    ANGLE --> DS
    ANGLE --> SECTOR[目标保护扇区]
    SBL --> INCM[INCM 重构]
    SECTOR --> INCM
    INCM --> MVDR[MVDR + iSTFT]

    H[目标头戴麦克风] --> M[对齐 SI-SDR]
    DS --> M
    MVDR --> M
```

其中 SBL-INCM-MVDR 的实现包括：

- 在 STFT 频点上用 MMV-SBL 估计稀疏空间功率；
- 排除视觉目标保护扇区，重构干扰加噪声协方差矩阵；
- 通过对角加载求解 MVDR 权重；
- 使用 iSTFT 恢复宽带语音。

对应代码在
[`src/cabin_speech/sbl_mvdr.py`](src/cabin_speech/sbl_mvdr.py)，论文公式与函数的对应关系记录在
[`docs/SBL_INCM_SPEECH.md`](docs/SBL_INCM_SPEECH.md)。

## 两个可运行入口

### 摄像头 + 普通麦克风实时演示

这部分是我在没有 USB 多通道阵列时做的实时交互版本。它使用摄像头、单麦克风、
能量 VAD、人脸跟踪和嘴部运动，显示 `SILENCE / SPEECH` 状态、活跃人脸和画外说话人，
还可以接入 Whisper 显示字幕。

```powershell
python -m cabin_speech demo
python -m cabin_speech demo --enable-asr
```

### AMI 八通道离线实验

真正的 DOA 和波束形成在同步八通道 AMI 数据上运行：

```powershell
python -m cabin_speech evaluate-active-speaker
python -m cabin_speech evaluate-interference
```

两条链路是有意分开的：实时入口展示视听交互，离线入口负责多通道阵列算法评测。

## 安装

项目使用 Python 3.11。我在 Windows 和 NVIDIA RTX 50 系列显卡环境中开发，建议先根据
[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装匹配显卡的 CUDA 版本，
再安装项目依赖：

```powershell
conda create -n cabin_avspeech python=3.11 pip -y
conda activate cabin_avspeech
python -m pip install -e ".[dev,demo,asr,evaluation]"
```

只运行测试和结果校验时，不需要下载 AMI 数据、YuNet 或 Light-ASD 权重：

```powershell
ruff check .
pytest
python -m cabin_speech verify-results
```

当前仓库包含 33 项自动测试。

## 数据准备

原始 AMI 数据、第三方代码和模型权重体积较大，没有提交到仓库。默认目录是：

```text
data/raw/ami/
├── ES2002a/
│   ├── audio/
│   │   ├── ES2002a.Array1-01.wav ... ES2002a.Array1-08.wav
│   │   └── ES2002a.Headset-0.wav ... ES2002a.Headset-3.wav
│   └── video/
│       └── ES2002a.Closeup1.avi ... ES2002a.Closeup4.avi
└── annotations/manual_1.6.2/
```

此外还需要：

- [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) 及其公开预训练权重；
- [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)；
- [`configs/ami_es2002a.yaml`](configs/ami_es2002a.yaml) 中的数据路径和人物方位角配置。

详细准备步骤和实验命令写在
[`docs/REAL_MULTIMODAL_AMI.md`](docs/REAL_MULTIMODAL_AMI.md)。

## 从哪里看代码

| 内容 | 入口 |
|---|---|
| YuNet 与 Light-ASD 适配 | [`active_speaker.py`](src/cabin_speech/active_speaker.py) |
| AMI 标注与人物映射 | [`annotations.py`](src/cabin_speech/annotations.py) |
| 多通道数据读取 | [`datasets.py`](src/cabin_speech/datasets.py) |
| SRP-PHAT | [`localization.py`](src/cabin_speech/localization.py) |
| SBL、INCM 与 MVDR | [`sbl_mvdr.py`](src/cabin_speech/sbl_mvdr.py) |
| 实时 VAD 与视听融合 | [`realtime.py`](src/cabin_speech/realtime.py) |
| 强干扰实验 | [`evaluate_ami_interference.py`](scripts/evaluate_ami_interference.py) |
| 完整模块和数据格式 | [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) |

## 关于实验范围

这是一次从数据解析、模型接入到阵列输出的完整链路实验，目前使用一场 AMI 会议和
16 个自动选取片段。强干扰样本来自两个真实房间片段的可控叠加；人物到方位角的关系
是针对 `ES2002a` 做的标定。实时版本使用单麦克风，八通道处理在离线数据上完成。

下一步我准备扩展到更多会议和自然重叠语音，并加入相机—阵列外参标定、不同 SIR/
角间隔曲线，以及 SBL 频点并行和收敛优化。

## 数据集与开源项目

- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [Light-ASD](https://github.com/Junhua-Liao/Light-ASD)
- [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [SBL-INCM-MVDR 论文](https://doi.org/10.1016/j.dsp.2026.106138)

第三方资源和许可信息见 [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)。
