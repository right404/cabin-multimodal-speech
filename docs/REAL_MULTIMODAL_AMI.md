# AMI 真实音视频多模态评测

## 数据链路

实验使用 AMI `ES2002a` 同一场会议中同步采集的：

- 8 通道圆形麦克风阵列；
- 4 路参会者近景视频；
- 4 路头戴麦克风参考信号；
- 人工转写、语音区间和参会者—摄像头—通道映射。

视觉前端使用 YuNet 和 Light-ASD；阵列前端使用 SRP-PHAT、Delay-and-Sum 和
SBL-INCM-MVDR。

## 主动说话人识别

程序从 XML 标注中自动选择时长 2–8 秒、有文本且无其他说话人重叠的区间。目标脸在
至少 50% 视频帧中被检测到时，记为目标可见。

| 评测范围 | 正确数 | Top-1 |
|---|---:|---:|
| 目标人物可见片段 | 9 / 10 | **90.00%** |
| 全部自动选取片段 | 9 / 16 | 56.25% |
| 四人随机选择参考 | — | 25.00% |

视觉覆盖率为 `10 / 16 = 62.50%`。结果 JSON 同时保存每个片段的预测人物、四路平均
说话概率、人脸检测率和目标可见状态：

```text
artifacts/ami/active_speaker_results.json
```

复现命令：

```powershell
python -m cabin_speech evaluate-active-speaker
```

## 视觉身份到阵列方向

强干扰实验从主动说话人结果读取目标人物，再查询固定座位方向：

```text
active_speaker_results.json
  -> 目标片段 prediction = B
  -> configs/ami_es2002a.yaml: B = 14°
  -> 视觉目标方向 = 14°
```

人物方向由 ES2002a 孤立标注语音的 SRP-PHAT 结果汇总得到。固定座位配置将视频中的人物
身份与阵列方位连接起来，使 Light-ASD 预测能够直接参与后续波束形成。

目标头戴麦克风通道由 `meetings.xml` 自动解析；本实验中人物 B 对应 `Headset-1`。

## +6 dB 强方向干扰

压力测试使用两个真实八通道房间片段构造空间混合：

- 目标：人物 B，`465.488–470.488 s`，方向 `14°`；
- 干扰：人物 D，`499.136–504.136 s`，方向 `−172°`；
- 干扰相对目标强 `6 dB`；
- 两段信号保留原会议室混响和八通道空间传递特性。

这是一个可重复的真实房间空间混合：所有前端处理同一个八通道输入，只改变目标方向来源。

| 对照项 | 结果 |
|---|---:|
| 纯音频 SRP-PHAT 目标方向 | `−172°` |
| Light-ASD 视觉目标方向 | `14°` |
| 视觉 Delay-and-Sum 相对纯音频方案 | **+2.20 dB SI-SDR** |
| 视觉 SBL-INCM-MVDR 相对纯音频方案 | **+2.15 dB SI-SDR** |

两种视觉引导后端都获得一致的相对收益，验证了视觉身份在强方向干扰下的目标选择作用。
完整绝对指标、方向结果和算法诊断保存在：

```text
artifacts/ami/interference_results.json
```

复现命令：

```powershell
python -m cabin_speech evaluate-interference
```

## SBL-INCM-MVDR 数据流

```text
8 通道空间混合
  -> STFT
  -> MMV-SBL 空间功率谱
  -> 视觉目标保护扇区
  -> 扇区外干扰峰筛选
  -> INCM 重构与对角加载
  -> MVDR 权重
  -> iSTFT 宽带语音
```

AMI 配置使用 `frequency_stride=2` 加速宽带处理；其余频点由同一视觉方向的频域
Delay-and-Sum 完成重建。结果 JSON 记录处理频点、迭代统计、保护扇区和最终指标，便于
检查算法是否进入最终输出。

## 数据路径

配置文件：

```text
configs/ami_es2002a.yaml
```

路径也可以通过命令行覆盖：

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

## 已打通模块

- AMI 八通道同步读取；
- XML 标注与参与者映射；
- YuNet + Light-ASD 四人主动说话人推理；
- 视觉身份到固定座位方向的数据链；
- SRP-PHAT 与 Delay-and-Sum；
- MMV-SBL、INCM 重构和 MVDR；
- 对头戴麦克风参考的时间对齐 SI-SDR；
- 结果 JSON 保存与自动校验。

## 来源

- AMI Corpus：<https://groups.inf.ed.ac.uk/ami/corpus/>
- AMI Signals：<https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml>
- Light-ASD：<https://github.com/Junhua-Liao/Light-ASD>
- YuNet：<https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- SBL-INCM-MVDR 论文：<https://doi.org/10.1016/j.dsp.2026.106138>
