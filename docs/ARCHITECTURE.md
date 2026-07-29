# 系统架构与代码数据流

## 1. 范围

仓库包含两条独立但共享部分算法模块的链路：

| 链路 | 输入 | 运行方式 | 主要输出 |
|---|---|---|---|
| 实时演示 | 普通摄像头、单麦克风 | 在线 | VAD、人脸/嘴部运动、画外说话人、可选 ASR |
| AMI 评测 | 4 路视频、8 通道阵列、4 路头戴麦克风、XML 标注 | 离线 | 主动说话人、DOA、波束形成、SI-SDR |

实时演示不运行阵列定位或波束形成；AMI 链路不代表已经实现实时八通道采集。

## 2. AMI 输入和标注

### 2.1 多通道音频

AMI 将阵列通道保存为 8 个同步单声道 WAV，而不是一个 8 通道文件。

`src/cabin_speech/datasets.py` 中：

- `load_synchronised_mono_files(paths, start_seconds, duration_seconds)`：
  - 验证所有文件均为单声道；
  - 验证采样率和总帧数一致；
  - 对每个文件执行相同的 sample seek；
  - 输出 `MultichannelRecording.audio`，形状为 `[channels, samples]`；
  - 当前 AMI 实验的数据类型为 `float64`，采样率为 16 kHz。
- `load_multichannel_audio` 用于单个多通道 WAV/FLAC，例如 AISHELL-4。

`MultichannelRecording` 同时携带 `sample_rate`、`source_path` 和时长属性。

### 2.2 XML 标注和参与者映射

`src/cabin_speech/annotations.py` 中：

- `read_ami_participants` 从 `corpusResources/meetings.xml` 解析：
  - AMI agent ID；
  - 头戴麦克风通道；
  - Closeup 摄像头编号；
  - global name 和会议角色。
- `read_ami_speech_intervals` 合并 `words/*.xml` 和 `segments/*.xml`，输出
  `SpeechInterval(speaker_id, start_seconds, end_seconds, text)`。
- `select_isolated_speech` 排除与其他说话人存在词汇区间重叠的片段。

主动说话人实验自动选择 2–8 秒的孤立片段，不在脚本中手工列出 16 个测试样本。

## 3. 视频处理

### 3.1 离线 YuNet 人脸裁剪

`src/cabin_speech/active_speaker.py::crop_face_sequence`：

1. 使用 OpenCV `FaceDetectorYN` 加载 YuNet ONNX；
2. 按毫秒定位到标注区间；
3. 每帧选择“检测置信度 × 人脸面积”最大的候选；
4. 将人脸框扩展为方形区域；
5. 转为灰度并缩放至 `112 × 112`；
6. 检测缺失时沿用上一个有效框；区间开始即缺失时填充黑帧；
7. 返回：
   - `face_frames`: `float` 推理前为 `uint8 [T, 112, 112]`；
   - `detection_rate`: 成功检测帧数 / 解码帧数。

`detection_rate` 只衡量 YuNet 是否检测到脸，不等价于“人一定正对相机”。当前实验用目标人物检测率 ≥ 0.5 定义目标可见子集。

### 3.2 实时 Haar 与嘴部帧差

`src/cabin_speech/vision.py::FaceMouthMotionTracker` 用于实时 Demo：

- Haar frontal-face detector；
- IoU 关联的轻量 track ID；
- 人脸下半部固定比例嘴部区域；
- 相邻嘴部图像绝对差和 EMA 平滑；
- 输出 `FaceObservation`。

`YuNetMouthMotionTracker` 是关键点对齐的实验性帧差实现，但 AMI 主结果使用
Light-ASD，而不是用嘴部帧差直接决定说话人。

## 4. Light-ASD 主动说话人识别

`src/cabin_speech/active_speaker.py::LightASDInference` 是对官方 Light-ASD 代码和权重的推理适配器：

1. 从第三方仓库导入 `ASD_Model` 和 `lossAV` 分类头；
2. 将持久化 state dict 按 `model.` 和 `lossAV.` 前缀拆分并加载；
3. 将阵列通道 1 的区间音频转为 16 kHz `int16`；
4. 使用 `python_speech_features.mfcc` 提取 13 维 MFCC，约 100 帧/秒；
5. 输入人脸灰度序列，约 25 帧/秒；
6. 截断到满足音频 4 帧对应视频 1 帧的共同长度；
7. 执行音频前端、视频前端、加法融合和双向 GRU 后端；
8. 输出每个视频帧的 audible-speaking 概率 `[T]`。

`scripts/evaluate_ami_active_speaker.py` 对每个标注区间使用同一段阵列音频，分别配对四路候选人脸序列，比较四个平均概率并输出 Top-1 人物身份。

结果保存为版本化 JSON：

```text
summary
segments[]
  start / end
  speaker / prediction / correct
  target_visible
  mean_probability{agent: probability}
  face_detection_rate{agent: rate}
```

## 5. 人脸身份到阵列方位角

当前实现不是从人脸框横坐标直接计算声学 DOA。

`configs/ami_es2002a.yaml::speaker_angles_deg` 保存 ES2002a 专用的身份—方向映射：

```text
A -> -134°
B ->   14°
C ->  -87°
D -> -172°
```

这些值来自孤立标注语音的经验 SRP-PHAT 方向汇总。会议中人物座位固定，因此主动说话人身份可映射到固定座位方向。

限制：

- 映射不能直接迁移到另一场会议；
- 没有使用相机内外参或阵列—相机联合标定；
- 当前实现是会议专用标定，还没有扩展为通用视觉 DOA 回归。

`scripts/evaluate_ami_interference.py` 使用
`read_active_speaker_prior` 读取主动说话人 JSON 的预测身份，然后查询该映射。这样视觉结果和最终波束形成之间有明确的数据血缘。

## 6. SRP-PHAT 声源定位

`src/cabin_speech/localization.py::estimate_srp_phat_doa` 输入：

- `multichannel_audio`: `[M, N]`；
- `sample_rate`；
- `CircularArrayGeometry`；
- 候选角度网格和 STFT 参数。

处理过程：

1. 对每个通道执行 SciPy STFT，得到 `[M, F, L]`；
2. 每个频点除以幅值，得到 PHAT 相位；
3. `steering_dictionary` 根据阵列几何构造 `[M, G]` 导向字典；
4. 在频率和快拍上累积 steered response；
5. 返回峰值角度、峰值相对中位数的启发式置信度及完整角谱。

强干扰实验中，SRP-PHAT 是纯音频目标选择基线。它在 +6 dB 干扰下选择 `-172°`，说明它被能量更强的 D 吸引；这不意味着定位算法没有找到声源，而是它不知道业务目标人物是谁。

## 7. Delay-and-Sum

`src/cabin_speech/audio.py::delay_and_sum`：

1. 通过 `CircularArrayGeometry.propagation_delays_s` 获得每个通道的远场时延；
2. `fractional_delay` 用线性插值完成亚采样延时；
3. 对时间对齐后的通道求均值；
4. 输出单通道波形 `[N]`。

声学基线使用 SRP-PHAT 角度；多模态方案使用视觉身份映射角度。两者处理相同的 8 通道混合信号。

## 8. SBL 空间谱

`src/cabin_speech/sbl_mvdr.py::run_mmv_sbl` 输入：

- 单个频点的阵列快拍 `snapshots [M, L]`；
- 角度导向字典 `dictionary [M, G]`；
- `SBLConfig`。

实现多快拍 Sparse Bayesian Learning：

- 迭代更新各角度超参数 `gamma [G]`；
- 更新噪声方差；
- 计算后验均值 `[G, L]`；
- 以行能量得到空间功率谱 `[G]`；
- 返回迭代数和 `converged` 标志。

## 9. INCM 重构

`reconstruct_incm_mvdr` 接收 SBL 功率谱、噪声方差、导向字典、角度网格和视觉目标角：

1. 在目标角附近创建保护扇区；
2. 保护扇区外查找局部峰；
3. 使用 `threshold_alpha × noise_variance` 筛选干扰峰；
4. 根据 `coverage_ratio` 选择覆盖主要能量的方向；
5. 使用选中导向矢量外积构造干扰协方差；
6. 加入噪声方差和对角加载，得到 loaded INCM。

`adaptive_sector_half_width` 将置信度映射为目标保护区宽度。这是项目在宽带语音场景中的扩展，不是原论文公式本身。

AMI 强干扰 pilot 为保持既有实验设置，使用配置中的固定 `beamforming_confidence=0.90`；
该值没有由 Light-ASD 原始概率完成校准。Light-ASD 真正决定的是目标身份和方向。

## 10. MVDR 权重和宽带重建

`reconstruct_incm_mvdr` 中的闭式解为：

```text
w = R_incm^-1 a_target / (a_target^H R_incm^-1 a_target)
```

`wideband_sbl_mvdr`：

1. 输入 `float64 [M, N]`；
2. STFT 得到 `[M, F, L]`；
3. 对选定频点独立执行 SBL、INCM 和 MVDR；
4. `conj(weights) @ spectra[:, bin, :]` 得到单通道频谱；
5. iSTFT 输出 `[N]`；
6. 返回处理频点、目标角、保护扇区、平均迭代数、收敛比例和处理频点比例。

当 `frequency_stride > 1` 时，未运行 SBL 的频点使用视觉方向的频域
Delay-and-Sum 权重。AMI 强干扰实验设置为 `frequency_stride=2`，所以结果是加速的 SBL/DS 混合宽带实现。

## 11. SI-SDR 评测

`src/cabin_speech/metrics.py::scale_invariant_sdr`：

- 去除均值；
- 将估计投影到参考方向；
- 比较投影能量与残差能量。

`scripts/evaluate_ami_interference.py::aligned_si_sdr` 先在 ±80 ms 范围内用互相关估计阵列输出和头戴麦克风之间的相对时延，再计算 SI-SDR。

参考是近讲头戴麦克风，估计是远场阵列输出，两者的传递函数不同。因此这里固定参考和
对齐方法，比较不同前端的相对 SI-SDR 变化。

## 12. 强干扰实验的数据血缘

```text
AMI XML
  -> 目标片段 B@465.488s / 干扰片段 D@499.136s

阵列 ch1 + 四路 Closeup
  -> YuNet + Light-ASD
  -> active_speaker_results.json: prediction=B

prediction=B
  -> ami_es2002a.yaml: B=14°
  -> visual target angle=14°

8ch B 片段 + 增益调整后的 8ch D 片段
  -> 同一 8ch 空间混合
  -> audio SRP target=-172°
  -> visual target=14°
  -> audio DS / visual DS / visual SBL-INCM-MVDR

B 的 Headset-1
  -> 时间对齐
  -> SI-SDR
```

两段阵列录音来自不同会议时间，因此这是受控的真实房间空间混合，不是原始自然重叠录音。

## 13. 实时链路

`src/cabin_speech/realtime.py`：

- `AudioLevelMonitor`：`sounddevice.InputStream` 单通道回调；
- `AdaptiveEnergyVAD`：启动噪声底校准、阈值和 hangover；
- `SpeechSegmenter`：pre-roll 和语音段拼接；
- `live_fusion_probabilities`：单麦克风语音概率 + 多脸嘴部运动的启发式融合。

`src/cabin_speech/asr.py`：

- `WhisperASR`：16 kHz 内存音频推理；
- `AsyncASRWorker`：单工作线程后台识别，避免阻塞视频循环。

实时视觉融合没有使用 `MultimodalFusionNet` 或 Light-ASD，也没有阵列 DOA。界面显示的 active 概率是启发式概率，不是 AMI 模型的校准输出。

## 14. 模块复用到雷达阵列的边界

可以迁移的算法抽象：

- `ArrayGeometry` 和导向矢量字典；
- 多通道/多阵元快拍矩阵；
- 稀疏角域 SBL；
- 干扰协方差重构；
- 对角加载和 MVDR；
- 阵列流形失配、有限快拍和强干扰实验设计。

尚未实现的雷达能力：

- 快时间/慢时间和脉冲数据立方体；
- 距离 FFT、多普勒 FFT；
- CFAR 检测；
- 波达方向与距离—速度联合估计；
- 航迹关联和目标跟踪；
- 真实雷达数据集验证。

因此可将该项目作为阵列信号处理能力证明，但不能称为完整雷达感知项目。
