# 视觉先验引导的宽带 SBL-INCM MVDR

## 方法来源

本模块以项目目录中的论文
`1-s2.0-S1051200426002575-main (1).pdf` 为窄带算法依据，复现：

- 式 (24)–(28)：MMV-SBL 后验与角功率谱；
- 式 (31)：目标扇区内的主导方向估计；
- 式 (34)：由 SBL 噪声方差校准的峰值筛选；
- 式 (39)–(41)：能量覆盖和方向—尺度解耦的干扰协方差重构；
- 式 (42)–(44)：INCM、对角加载和 MVDR 权重；
- 附录 A 的残差型噪声方差更新，而不是式 (A.8) 的迹修正备选式。

论文原方法处理窄带阵列快拍。本项目通过 STFT 将它迁移到宽带语音，每个频点独立
估计 SBL 空间谱和稳健 MVDR 权重，再经 iSTFT 还原语音。

## 从窄带算法到语音场景

摄像头给出目标说话人的粗方向和视觉置信度。置信度越高，目标保护扇区越窄；遮挡、
侧脸或唇动不稳定时，保护扇区自动变宽，降低目标语音泄漏到干扰协方差中的风险。

“视觉置信度自适应保护扇区”是我在语音场景中增加的设计，原论文中没有这一部分。
代码里把论文步骤和新增逻辑分开标注，方便逐段检查。

需要区分设计能力和已验证设置：AMI 强干扰 pilot 的目标身份来自 Light-ASD，但保护
扇区置信度固定为配置中的 `0.90`，尚未由 Light-ASD 原始概率完成校准。

## 代码映射

- `run_mmv_sbl`：附录 A 的 MMV-SBL；
- `reconstruct_incm_mvdr`：论文算法 1 的步骤 3–5；
- `adaptive_sector_half_width`：视觉扩展；
- `wideband_sbl_mvdr`：STFT 宽带化和 iSTFT 重建；
- `scripts/run_sbl_mvdr_demo.py`：可复现的多声源仿真与 SI-SDR 评测。

当 `frequency_stride > 1` 时，仅选定频点运行 SBL-INCM-MVDR，其他频点回退到视觉
Delay-and-Sum。AMI 实验使用 `frequency_stride=2`；实际 SBL 处理约 20.23% 的全部
STFT bins，所有被处理频点都运行到了 80 次迭代上限。结果文件保留了这一诊断信息，
后续还需要继续调整初始化和停止条件。

## 快速验证

```powershell
conda activate cabin_avspeech
python scripts/run_sbl_mvdr_demo.py
```

输出在 `artifacts/sbl_mvdr_demo/`：

- `reference.wav`
- `mixture.wav`
- `delay_sum.wav`
- `sbl_incm_mvdr.wav`
- `metrics.json`

快速模式使用较稀的角度网格和较少迭代次数以控制等待时间。按论文默认的 1° 网格和 200 次最大
迭代运行：

```powershell
python scripts/run_sbl_mvdr_demo.py --paper-settings
```

## 后续实验矩阵

在公开多通道音视频数据上至少报告：

| 方法 | 视觉先验 | 稳健 INCM | 指标 |
|---|---:|---:|---|
| 单通道原始音频 | 否 | 否 | SI-SDR、STOI、PESQ、CER |
| Delay-and-Sum | 粗方向 | 否 | 同上 |
| SCM-MVDR | 可选 | 否 | 同上 |
| 论文 SBL-INCM MVDR | 固定 5° 扇区 | 是 | 同上 |
| 本项目方法 | 置信度自适应扇区 | 是 | 同上 + RTF |

摄像头加普通麦克风的实时程序用于演示 VAD、人脸/唇动和 ASR；AMI 离线链路使用同步
8 通道数据进行波束形成。当前实验聚焦视觉目标消歧，两种视觉引导后端在这组数据上
取得了接近的结果。真实多模态结果见
[REAL_MULTIMODAL_AMI.md](REAL_MULTIMODAL_AMI.md)。
