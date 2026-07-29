# AMI 真实音视频多模态评测

## 结论

项目已经接入同一场 AMI 会议中同步采集的：

- 8 通道圆形麦克风阵列；
- 4 路参会者近景视频；
- 4 路头戴麦克风参考信号；
- 人工转写、语音区间和参会者—摄像头—通道映射。

视觉前端使用 YuNet 和 Light-ASD；阵列前端使用 SRP-PHAT、Delay-and-Sum，以及项目复现的 SBL-INCM-MVDR。

当前 pilot experiment 的主要证据是：当强方向干扰让纯音频方法选错人物时，视觉主动说话人身份能够纠正目标方向，并给后续波束形成带来约 2.2 dB 的相对 SI-SDR 收益。两种视觉引导后端的结果接近，实验重点是验证目标身份消歧。

## 数据与切分

- 数据：AMI Meeting Corpus，会议 `ES2002a`；
- 自动从 XML 读取四名参会者与摄像头/头戴麦克风映射；
- 自动选择时长 2–8 秒、有文本且不与其他说话人重叠的区间；
- 共得到 16 个区间，不手工挑选识别正确的样本；
- 目标脸检测率至少 50% 时定义为“目标可见”，共 10 个区间。

近景摄像头有时会拍向白板，目标人物完全离开画面。因此同时报告全量结果、可见子集结果和视觉覆盖率，不能只报告可见子集准确率。

所有会议专用参数位于 `configs/ami_es2002a.yaml`。大型数据、Light-ASD 第三方代码、模型权重和 YuNet ONNX 均不提交到仓库。

## 主动说话人识别

| 评测集合 | 正确数 | Top-1 准确率 |
|---|---:|---:|
| 全部自动选取区间 | 9 / 16 | 56.25% |
| 目标可见区间 | 9 / 10 | 90.00% |
| 四人随机选择参考 | — | 25.00% |

视觉覆盖率为 `10 / 16 = 62.50%`。`90%` 只适用于目标人物可见的 10 个片段。主要错误来源是侧脸或人物离开画面；目标不在视频中时不能从该路视觉信息恢复。

结果：

- `artifacts/ami/active_speaker_results.json`

复现：

```powershell
python -m cabin_speech evaluate-active-speaker
```

## 视觉先验到阵列方向的数据链

强干扰脚本不再直接写死“视觉目标为 14°”，而是执行：

```text
active_speaker_results.json
  -> 查找 465.488 s 目标片段
  -> Light-ASD prediction = B
  -> configs/ami_es2002a.yaml: B = 14°
  -> 视觉目标方向 = 14°
```

身份—角度表是从 ES2002a 孤立标注语音得到的经验方向标定，只适用于该会议，不是从人脸像素坐标直接估计 DOA 的通用模型。

目标头戴麦克风通道由 `meetings.xml` 解析；本实验中 B 对应 `Headset-1`。

## +6 dB 强方向干扰压力测试

使用两个不同时间的真实 AMI 阵列片段构造空间混合：

- 目标：说话人 B，`465.488–470.488 s`，方向 `14°`；
- 干扰：说话人 D，`499.136–504.136 s`，方向 `-172°`；
- 干扰相对目标强 `6 dB`；
- 两段信号都保留真实会议室混响和 8 通道空间传递特性。

这不是原会议中自然发生的重叠语音，而是由两个真实多通道房间录音构造的可控压力测试。

纯音频 SRP-PHAT 被更强干扰带到 `-172°`；持久化的 Light-ASD 结果选择 B，因此视觉链路使用 `14°`。

| 前端 | 对目标头戴麦克风的对齐 SI-SDR | 相对纯音频定位 |
|---|---:|---:|
| 单通道 | -12.37 dB | -0.06 dB |
| 纯音频 DOA + Delay-and-Sum | -12.31 dB | 基线 |
| 视觉目标 + Delay-and-Sum | -10.11 dB | **+2.20 dB** |
| 视觉目标 + 加速 SBL-INCM-MVDR | -10.17 dB | **+2.15 dB** |

绝对 SI-SDR 为负与“远场阵列输出对近讲头戴麦克风参考”的跨设备、跨传递函数比较有关；这里主要比较同一参考、同一对齐方式下的相对变化。

结果：

- `artifacts/ami/interference_results.json`

复现：

```powershell
python -m cabin_speech evaluate-interference
```

## SBL 实际运行状态

结果 JSON 记录了：

- `max_iterations = 80`；
- `mean_iterations = 80.0`；
- `converged_fraction = 0.0`；
- `frequency_stride = 2`；
- SBL 实际处理频点占全部 STFT bins 的约 `20.23%`；
- 未处理频点回退到视觉 Delay-and-Sum；
- 保护扇区半宽为 `5.1°`。

因此该结果是加速的 SBL/DS 混合宽带实现，并且当前停止准则下所有被处理频点都达到迭代上限，不能宣称 SBL 已经稳定收敛。

当前单个压力案例不足以对两种波束形成后端作普适排序。能够确认的是：

1. `wideband_sbl_mvdr` 确实被调用；
2. 视觉身份映射角确实进入目标保护扇区；
3. SBL 空间谱、INCM 重构和 MVDR 权重参与了被选频点的最终重建；
4. 主要增益来自视觉纠正目标选择，而不是 SBL 后端优于 DS。

实验为保持原 pilot 数字，使用受控 `beamforming_confidence=0.90` 调节保护扇区。目标片段的 Light-ASD 原始平均概率约为 `0.322`，尚未进行概率校准，所以不能把 0.90 描述为模型直接输出的置信度。

## 复现性说明

配置：

- `configs/ami_es2002a.yaml`

路径可通过命令行覆盖：

```powershell
python scripts/evaluate_ami_active_speaker.py `
  --ami-root D:/datasets/ami `
  --annotation-root D:/datasets/ami/annotations/manual_1.6.2 `
  --light-asd-root D:/models/Light-ASD `
  --yunet-model D:/models/face_detection_yunet_2023mar.onnx

python scripts/evaluate_ami_interference.py `
  --audio-root D:/datasets/ami/ES2002a/audio `
  --active-speaker-results artifacts/ami/active_speaker_results.json
```

Windows 上部分 OpenCV 版本不能从包含中文字符的绝对路径读取 ONNX。`crop_face_sequence` 会在必要时将 YuNet 暂存到 ASCII 临时目录，再交给 OpenCV；源模型仍保留在配置路径。

## 能力边界

已经验证：

- 同步读取 8 个独立阵列 WAV；
- AMI 标注和参与者映射解析；
- YuNet + Light-ASD 四人主动说话人推理；
- 持久化视觉身份到阵列方向的数据链；
- SRP-PHAT、Delay-and-Sum 和加速 SBL-INCM-MVDR；
- 对头戴麦克风参考的对齐 SI-SDR；
- 单麦克风摄像头实时 Demo。

尚未验证：

- 完整 AMI benchmark 或跨会议泛化；
- 自然重叠语音上的同等增益；
- 通用相机坐标到阵列 DOA；
- Light-ASD 概率到保护扇区的可靠校准；
- SBL 在该真实数据设置下的稳定收敛和优于 DS；
- 实时 8 通道采集与波束形成；
- 雷达距离—多普勒、检测或跟踪。

## 求职建议表述

> 搭建 AMI 真实会议视听阵列前端，接入同步 8 通道圆阵、4 路视频、4 路头戴麦克风及人工标注，融合 YuNet、Light-ASD、SRP-PHAT 与 SBL-INCM-MVDR；在目标人物可见的 10 个自动选取片段上实现 9/10 主动说话人 Top-1，并在 +6 dB 强方向干扰下通过视觉身份消歧，使 DS/SBL 相对纯音频定位方案分别提升 2.20/2.15 dB SI-SDR，同时报告 62.5% 视觉覆盖率和 SBL 收敛局限。

面试时应主动说明：总样本量只有 16 段，可见子集 10 段；这是完成端到端链路验证的 pilot experiment，不是通用 SOTA benchmark。

## 来源

- AMI Corpus：<https://groups.inf.ed.ac.uk/ami/corpus/>
- AMI Signals：<https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml>
- Light-ASD：<https://github.com/Junhua-Liao/Light-ASD>
- YuNet：<https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- SBL-INCM-MVDR 论文：<https://doi.org/10.1016/j.dsp.2026.106138>
