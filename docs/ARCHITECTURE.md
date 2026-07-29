# 系统架构与代码数据流

## 1. 三条运行链路

| 链路 | 输入 | 运行方式 | 输出 |
|---|---|---|---|
| 实时演示 | 普通摄像头、单麦克风 | 在线 | VAD、人脸/嘴部运动、画外说话人、可选 ASR |
| AMI 多模态评测 | 4 路视频、8 通道阵列、4 路头戴麦克风、XML 标注 | 离线 | 主动说话人、DOA、波束形成、SI-SDR |
| AISHELL-4 阵列评测 | 8 通道圆阵、TextGrid/RTTM | 离线 | DOA、波束形成、ASR CER |

实时入口复用音频分段和 ASR 等基础模块；AMI 与 AISHELL-4 共用阵列几何、定位、
波束形成和指标模块。

## 2. AMI 多通道输入

AMI 阵列录音保存为 8 个同步单声道 WAV。`src/cabin_speech/datasets.py` 中的
`load_synchronised_mono_files`：

1. 验证通道数、采样率和总帧数；
2. 对每个文件执行相同的 sample seek；
3. 按 `[channels, samples]` 堆叠；
4. 返回 `MultichannelRecording`，同时携带采样率、来源路径和时长。

`load_multichannel_audio` 用于读取单个多通道 WAV/FLAC，例如 AISHELL-4。

## 3. 标注与片段选择

`src/cabin_speech/annotations.py`：

- `read_ami_participants` 从 `meetings.xml` 解析人物、头戴麦克风通道、近景摄像头和角色；
- `read_ami_speech_intervals` 合并 word/segment XML，生成
  `SpeechInterval(speaker_id, start_seconds, end_seconds, text)`；
- `select_isolated_speech` 自动筛选 2–8 秒无重叠语音片段。

测试片段由标注条件自动生成，结果 JSON 保存片段起止时间和人物映射。

## 4. 视频与主动说话人

### 4.1 YuNet 人脸序列

`src/cabin_speech/active_speaker.py::crop_face_sequence`：

1. 使用 OpenCV `FaceDetectorYN` 加载 YuNet；
2. 定位到标注时间区间；
3. 每帧选择“检测置信度 × 人脸面积”最大的候选；
4. 扩展方形区域并缩放为 `112 × 112` 灰度人脸；
5. 使用上一有效框保持短时序列连续；
6. 返回人脸帧和检测率。

### 4.2 Light-ASD

`LightASDInference` 对官方模型和权重进行推理适配：

1. 从第三方仓库导入 `ASD_Model` 与 `lossAV`；
2. 加载音频、视频和融合分类头参数；
3. 将阵列通道 1 转换为 16 kHz `int16`；
4. 提取 13 维 MFCC，约 100 帧/秒；
5. 将音频与约 25 帧/秒的人脸序列按 4:1 对齐；
6. 执行音频前端、视频前端、融合层和双向 GRU；
7. 输出每个视频帧的 audible-speaking 概率。

主动说话人脚本将同一音频分别与四路候选人脸配对，按平均说话概率选出 Top-1 人物。

## 5. 人物身份到阵列方位

`configs/ami_es2002a.yaml::speaker_angles_deg` 保存固定座位方向：

```text
A -> -134°
B ->   14°
C ->  -87°
D -> -172°
```

这些方向由 ES2002a 孤立标注语音的 SRP-PHAT 结果汇总得到。主动说话人身份查询座位配置
后生成 `visual_target_angle_deg`，供 Delay-and-Sum 和目标保护扇区使用。

## 6. SRP-PHAT

`src/cabin_speech/localization.py::estimate_srp_phat_doa` 输入多通道波形、采样率、阵列几何
和候选角度网格：

1. 对各通道执行 STFT；
2. 对每个频点做 PHAT 幅值归一化；
3. 根据 `ArrayGeometry` 构造角度导向字典；
4. 在频率和快拍维累积 steered response；
5. 输出峰值方向、角谱和峰值置信度。

在强干扰实验中，SRP-PHAT 作为纯音频目标选择对照；视觉身份提供业务目标方向。

## 7. Delay-and-Sum

`src/cabin_speech/audio.py::delay_and_sum`：

1. 由阵列几何计算远场传播时延；
2. `fractional_delay` 通过线性插值完成亚采样延时；
3. 对齐后对各通道求均值；
4. 输出单通道增强波形。

声学方案使用 SRP-PHAT 方向，视觉方案使用人物身份映射方向，两者处理同一八通道输入。

## 8. MMV-SBL 空间谱

`src/cabin_speech/sbl_mvdr.py::run_mmv_sbl` 输入单频点快拍矩阵 `[M, L]` 和角度导向字典
`[M, G]`：

- 迭代更新角度超参数 `gamma [G]`；
- 估计噪声方差；
- 计算后验均值 `[G, L]`；
- 使用行能量生成空间功率谱 `[G]`；
- 返回角谱、迭代数和收敛状态。

## 9. INCM 重构与 MVDR

`reconstruct_incm_mvdr`：

1. 根据目标方向和置信度建立保护扇区；
2. 在扇区外搜索局部峰；
3. 使用 SBL 噪声方差筛选干扰方向；
4. 按能量覆盖率选择主要干扰；
5. 用导向矢量外积重构干扰协方差；
6. 加入噪声项与对角加载；
7. 求解目标方向无失真的 MVDR 权重。

MVDR 闭式解：

```text
w = R_incm^-1 a_target / (a_target^H R_incm^-1 a_target)
```

`wideband_sbl_mvdr` 在 STFT 频点上执行上述步骤，并使用 iSTFT 输出时域波形。配置项
`frequency_stride` 用于在计算量和频率分辨率之间切换。

## 10. SI-SDR 与 CER

`src/cabin_speech/metrics.py::scale_invariant_sdr`：

- 去除均值；
- 将估计信号投影到参考方向；
- 比较投影能量和残差能量。

AMI 评测先通过互相关估计阵列输出与头戴麦克风之间的相对时延，再计算对齐 SI-SDR。

AISHELL-4 评测将增强语音送入 Whisper，统一文本格式后计算中文 CER，并保存逐句替换、
删除和插入统计。

## 11. 强干扰实验数据血缘

```text
AMI XML
  -> 目标片段 B@465.488s
  -> 干扰片段 D@499.136s

阵列 ch1 + 四路 Closeup
  -> YuNet + Light-ASD
  -> active_speaker_results.json: prediction=B

prediction=B
  -> ami_es2002a.yaml: B=14°
  -> visual target angle=14°

同一 8ch 空间混合
  -> audio SRP target=-172°
  -> visual target=14°
  -> audio DS / visual DS / visual SBL-INCM-MVDR

B 的 Headset-1
  -> 时间对齐
  -> SI-SDR
```

## 12. 实时链路

`src/cabin_speech/realtime.py`：

- `AudioLevelMonitor`：单通道音频回调；
- `AdaptiveEnergyVAD`：噪声底校准、阈值和 hangover；
- `SpeechSegmenter`：pre-roll 与语音段拼接；
- `FaceMouthMotionTracker`：人脸跟踪和嘴部运动；
- `live_fusion_probabilities`：语音状态与多脸嘴部运动融合。

`src/cabin_speech/asr.py`：

- `WhisperASR`：16 kHz 内存音频推理；
- `AsyncASRWorker`：后台线程识别，保持视频循环响应。

## 13. 可复用的阵列处理模块

项目中的以下抽象可以直接用于音频阵列、通信阵列和雷达阵列原型：

- 二维阵列坐标与导向矢量字典；
- 多阵元快拍矩阵；
- 宽带 STFT 分频处理；
- SRP-PHAT 和稀疏角谱；
- 干扰协方差重构；
- 对角加载与 MVDR；
- 有限快拍、强干扰和方向失配实验设计。
