# 求职展示材料

## 使用原则

- `90%` 必须写成“目标人物可见的 10 个片段上 9/10”，不能写成完整 AMI 准确率；
- 同时说明全部 16 个自动选取片段为 9/16、视觉覆盖率为 62.5%；
- 当前证据聚焦视觉目标消歧，不对 SBL-INCM-MVDR 与 Delay-and-Sum 作普适排序；
- `+2.15 dB` 和 `+2.20 dB` 都是相对“纯音频 DOA + Delay-and-Sum”基线；
- 强干扰数据是两个真实多通道房间片段的可控叠加，不是自然同步重叠语音；
- 实时演示为单麦克风，8 通道波束形成是离线评测。

---

# 中文简历版本

## 一行项目简介

构建基于 AMI 同步 4 路视频与 8 通道圆阵的目标说话人语音前端，以 Light-ASD 视觉身份先验解决强方向干扰下的目标消歧，并引导 SRP-PHAT、Delay-and-Sum 与 SBL-INCM-MVDR。

## 3 条简历项目要点

- 搭建真实视听阵列评测链路，解析 AMI `ES2002a` 的 8 通道圆阵、4 路近景视频、4 路头戴麦克风及 XML 标注，完成同步分段、参与者映射和可复现 JSON 结果保存。
- 集成 YuNet 与 Light-ASD 完成四人主动说话人识别，在目标人物可见的 10 个自动选取片段上实现 `9/10` Top-1；同时报告全部 16 段 `9/16` 和 `62.5%` 视觉覆盖率并分析离画/侧脸失败。
- 实现 SRP-PHAT、Delay-and-Sum 及宽带 SBL-INCM-MVDR；在干扰比目标强 `6 dB`、纯音频定位锁定错误人物的可控真实房间混合中，视觉引导 DS 和 SBL 方案分别提升 `2.20 dB`、`2.15 dB` SI-SDR。

## 5 条详细简历项目要点

- 设计真实会议场景数据流水线，按采样点同步读取 AMI 8 个独立单声道阵列文件，并从 `meetings.xml`、word/segment XML 自动恢复人物—摄像头—头戴麦克风映射及无重叠语音区间。
- 接入 OpenCV YuNet 人脸检测和 CVPR 2023 Light-ASD 音视频融合模型，将 100 fps MFCC 与 25 fps 人脸序列对齐，对四路候选视频执行主动说话人 Top-1 推理。
- 建立覆盖率审计与失败分析：全部 16 个自动片段准确率 `56.25%`，目标脸检测率 ≥50% 的 10 个片段准确率 `90%`，视觉覆盖率 `62.5%`，避免将可见子集结果误报为完整数据集性能。
- 基于 8 通道圆阵实现 SRP-PHAT DOA、分数延时 Delay-and-Sum、MMV-SBL 角谱、噪声校准干扰筛选、INCM 重构、对角加载 MVDR 和 STFT/iSTFT 宽带重建。
- 构造 `+6 dB` 强方向干扰压力测试并打通“Light-ASD 身份 → 会议专用角度标定 → 波束形成”链路；纯音频 DOA 从目标 `14°` 被带偏至干扰 `−172°` 时，多模态 DS/SBL 相对纯音频方案分别提升 `2.20/2.15 dB` SI-SDR，验证收益来自视觉目标纠正。

---

# English Resume Version

## One-line project summary

Built an audio-visual target-speaker front end on synchronized AMI four-camera and eight-channel circular-array data, using Light-ASD identity priors to disambiguate the desired speaker under strong directional interference and guide SRP-PHAT, delay-and-sum, and SBL-INCM-MVDR beamforming.

## Three resume bullets

- Built a reproducible multimodal array-processing pipeline for AMI `ES2002a`, integrating an eight-channel circular array, four close-up videos, four headset references, XML annotations, synchronized segment loading, and versioned JSON evaluation outputs.
- Integrated YuNet and Light-ASD for four-way active-speaker identification, achieving `9/10` Top-1 accuracy on automatically selected segments where the target face was visible; also reported `9/16` on all segments and `62.5%` visual coverage with failure analysis.
- Implemented SRP-PHAT localization, delay-and-sum, and wideband SBL-INCM-MVDR; in a controlled real-room mixture with a directional interferer `6 dB` stronger than the target, visual guidance improved SI-SDR by `2.20 dB` and `2.15 dB`, respectively, over audio-only DOA beamforming.

## Five detailed resume bullets

- Developed a synchronized AMI data pipeline that stacks eight mono array recordings by sample offset and parses meeting, word, and segment XML files into speaker, camera, headset-channel, timing, and transcript mappings.
- Adapted the CVPR 2023 Light-ASD model with YuNet face crops, aligning 100-fps MFCC features with 25-fps face sequences and scoring four candidate participants for each isolated utterance.
- Added visibility-aware evaluation and failure auditing: `56.25%` Top-1 on all 16 automatically selected segments, `90%` on the 10 target-visible segments, and `62.5%` visual coverage, explicitly separating subset performance from corpus-level claims.
- Implemented an eight-channel signal-processing stack covering SRP-PHAT, fractional-delay delay-and-sum, MMV sparse Bayesian spatial-spectrum estimation, interference-plus-noise covariance reconstruction, diagonal loading, MVDR weights, and STFT/iSTFT synthesis.
- Connected persisted Light-ASD identity decisions to an ES2002a-specific azimuth calibration; when audio-only localization switched from the `14°` target to the `−172°` interferer at `+6 dB`, visual DS and visual SBL improved SI-SDR by `2.20/2.15 dB`, validating the contribution of visual target correction.

---

# 面试介绍

## 30 秒介绍

这个项目解决的是“阵列知道最强声音从哪来，但不知道业务上应该听谁”的问题。我用 AMI 一场真实会议的 4 路视频、8 通道圆阵和人工标注，先用 YuNet 和 Light-ASD 判断目标说话人，再把身份映射到座位方向，引导 Delay-and-Sum 和 SBL-INCM-MVDR。在一个干扰比目标强 6 dB 的压力测试中，纯音频定位被带到错误人物，视觉引导方案相对纯音频基线提升了约 2.2 dB SI-SDR。当前是小规模 pilot，不是完整 AMI benchmark。

## 1 分钟介绍

我把项目分成实时展示和离线评测两条链路。实时部分只使用普通摄像头和单麦克风，完成 VAD、人脸嘴部运动、画外说话人提示和可选 Whisper 字幕；因为没有阵列，所以不输出虚假 DOA，也不运行 MVDR。

离线部分使用 AMI `ES2002a` 的同步 4 路视频、8 通道圆阵、4 路头戴麦克风和 XML 标注。视频经过 YuNet 裁脸，与阵列一路音频一起送入 Light-ASD，得到四人主动说话人预测。目标人物可见的 10 个自动片段上是 9/10，但全部 16 段只有 9/16，主要问题是人物离开近景画面，视觉覆盖率只有 62.5%。

我进一步把 Light-ASD 预测身份映射到固定座位角，引导 8 通道波束形成。在 +6 dB 强干扰下，SRP-PHAT 选择了错误人物，而视觉先验仍选择正确目标，视觉 DS 和 SBL 分别比纯音频定位方案提升 2.20 和 2.15 dB。两种后端在当前案例中的结果接近，核心结论是视觉完成了目标纠正。

## 3 分钟技术介绍

项目的出发点是把“声源定位”和“目标选择”分开。SRP-PHAT 可以定位空间中能量较强的声源，但在多人座舱或会议中，最强声源不一定是系统想增强的人。视觉可以提供人物身份和说话状态，因此适合作为语义层先验。

数据侧，我使用 AMI `ES2002a`。阵列数据是 8 个独立的 16 kHz 单声道 WAV，我实现了按相同 sample offset 的同步读取，并验证采样率和帧数一致。标注侧解析 `meetings.xml` 获得 agent、摄像头和头戴麦克风通道映射，再合并 word/segment XML，自动选择 2 到 8 秒且没有其他说话人重叠的区间。

视觉侧使用 YuNet 逐帧检测并生成 `112×112` 灰度人脸序列。Light-ASD 的音频输入是阵列第一通道的 13 维 MFCC，约 100 fps；视频约 25 fps，按四比一对齐。对同一音频分别与四路候选人脸配对，比较平均 audible-speaking 概率，得到人物身份。全部 16 个片段正确 9 个；目标脸检测率至少 50% 的 10 个片段正确 9 个。这里我同时报告覆盖率，因为人物不在画面中时，模型不能凭空恢复视觉信息。

阵列侧使用 8 通道圆阵模型。SRP-PHAT 对各频点做相位归一化并扫描角度字典。Delay-and-Sum 用几何传播时延完成分数延时对齐。SBL 部分在每个选定 STFT 频点把阵列快拍表示成稀疏角域模型，估计角功率和噪声方差；INCM 重构排除视觉目标保护扇区，在扇区外筛选主要干扰峰，合成干扰加噪声协方差，再通过对角加载求 MVDR 权重并 iSTFT。

为了验证视觉先验的因果作用，我把 B 和 D 两个不同时间的真实 8 通道房间片段叠加，并把 D 调到比 B 强 6 dB。纯音频 SRP-PHAT 从 B 的 14° 跳到了 D 的 −172°；Light-ASD 的持久化结果仍选择 B，因此多模态链路使用 14°。相对纯音频 DOA 加 DS，视觉 DS 提升 2.20 dB，视觉 SBL 提升 2.15 dB。这说明收益主要来自正确的目标选择，不是来自声称 SBL 全面更强。

当前局限包括：只测了一场会议和 16 个孤立片段；身份—角度是会议专用标定；干扰是可控叠加而非自然重叠；SBL 以 stride 2 处理频点且在 pilot 参数下达到迭代上限；实时链路仍是单麦克风。下一步我会扩展会议数量、做自然重叠语音和阵列失配实验，加入相机—阵列外参标定，并优化 SBL 收敛与 GPU 并行。

---

# 常见追问

## 1. 为什么要做多模态？

音频提供“哪里有声源”和语音内容，视觉提供“哪个人物正在发声”的身份证据。强干扰下，音频定位可能正确找到最强声源，却不是业务目标；视觉可以在目标选择层消除这个歧义。项目的 +6 dB 实验就是为验证这一点设计的。

## 2. 为什么纯音频会选错人？

当前纯音频基线按 SRP-PHAT 峰值选择方向。干扰者比目标强 6 dB 后，`−172°` 方向的空间响应超过目标 `14°`，峰值选择自然切到干扰者。算法找到的是更强声源，但没有人物身份或业务目标信息。

## 3. 视觉信息如何进入波束形成？

Light-ASD 先输出人物身份。由于 ES2002a 中座位固定，配置文件将人物身份映射到经验方位角。Delay-and-Sum 直接使用该角度计算传播延时；SBL-INCM-MVDR 则用该角度建立目标保护扇区，避免把目标能量写入干扰协方差。当前不是从人脸像素直接回归角度。

## 4. SBL-INCM-MVDR 做了什么？

MMV-SBL 在角度字典上从多通道快拍估计稀疏空间功率和噪声方差。随后在视觉目标保护扇区外筛选主要干扰方向，用导向矢量外积重构 INCM，加入噪声项和对角加载，再求满足目标方向无失真的最小方差权重。宽带语音通过 STFT 分频处理并由 iSTFT 重建。

## 5. 如何理解 SBL-INCM-MVDR 与 Delay-and-Sum 的结果？

当前只有一个构造的强干扰 pilot，阵列方向采用经验标定，目标参考来自不同传递路径的头戴麦克风；SBL 还使用 stride 2 加速，未处理频点回退到 DS，并在当前配置下达到迭代上限。两种后端结果接近，现有样本不足以作普适排序。项目当前验证的是视觉先验能够把目标方向从干扰者纠正回目标人物。

## 6. 如何证明提升来自视觉先验？

压力测试中所有方法使用完全相同的 8 通道混合和参考信号，只改变目标角来源。纯音频 SRP-PHAT 选择 `−172°`，而持久化的 Light-ASD 结果选择人物 B，并映射到 `14°`。视觉 DS 与视觉 SBL 都比使用错误声学方向的基线高约 2.2 dB，而视觉 SBL 与视觉 DS 很接近，说明主要收益来自目标角纠正，而不是 SBL 后端本身。

更严格的后续证明应加入多场会议、多 SIR 条件、oracle 角度上限、随机错误视觉先验和置信区间。

## 7. 实时链路为什么只有单麦克风？

开发电脑目前没有 USB 多通道阵列。实时 Demo 的目标是验证摄像头、VAD、嘴部运动和 ASR 交互，因此明确限制为单麦克风。真正的 DOA 和波束形成只能在 AMI 的同步 8 通道数据上离线验证，不能把两条链路合并描述成实时八通道系统。

## 8. 这个项目如何迁移到雷达阵列处理？

可迁移的是阵列信号处理抽象：阵列几何和导向矢量、多阵元快拍、稀疏角谱 SBL、有限快拍条件下的协方差估计、INCM 重构、对角加载和 MVDR。语音中的 STFT 时间—频率单元可以类比雷达的快拍或距离—多普勒单元。

但项目没有实现距离 FFT、多普勒 FFT、CFAR、点迹或跟踪，也没有雷达数据验证。因此只能证明相关阵列算法基础，不能声称已经完成雷达感知系统。

## 9. 当前项目最大的局限是什么？

最大的实验局限是样本规模和视觉覆盖：只有一场会议、16 个孤立片段，目标可见的只有 10 个。最大的工程局限是身份—方位角仍依赖 ES2002a 专用标定，实时端没有多通道硬件。最大的算法局限是 SBL 尚未在多场景中稳定超过简单基线。

## 10. 后续如何扩展？

优先顺序建议：

1. 扩展到更多 AMI meeting 和自然重叠语音，按会议划分训练/验证/测试；
2. 报告不同 SIR、混响、目标—干扰角间隔、视觉遮挡条件的曲线和置信区间；
3. 加入 oracle angle、audio-only、visual-only、错误视觉先验和置信度门控消融；
4. 使用相机标定、座位检测或多视角几何替代会议专用身份—角度表；
5. 校准 Light-ASD 概率并研究视觉置信度到保护扇区宽度的映射；
6. 优化 SBL 初始化、停止条件和频点 GPU 并行，报告收敛率与 RTF；
7. 接入真实 USB 阵列，实现实时多通道采集、同步和在线波束形成；
8. 若投递雷达岗，增加公开 FMCW/MIMO 雷达数据上的距离—多普勒—DOA 最小实验。

---

# 推荐选择

中文简历优先使用“3 条简历项目要点”。如果版面允许，在项目标题后补充：

> Python / PyTorch / OpenCV / SciPy / Array Signal Processing / Audio-Visual Learning

英文简历优先使用“Three resume bullets”。不要把 `pilot experiment` 改写为
`state-of-the-art benchmark`，也不要省略 `target-visible segments` 限定。
