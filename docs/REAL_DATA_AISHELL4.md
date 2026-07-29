# AISHELL-4 真实多通道实验

## 数据

数据来自 [OpenSLR SLR111](https://www.openslr.org/111/)，许可为 CC BY-SA 4.0。
当前下载的是约 5.2 GB 的 test 压缩包，解压后包含：

- 20 场真实中文会议；
- 16 kHz、8 通道圆形麦克风阵列 FLAC；
- 逐说话人的 Praat TextGrid；
- RTTM 说话人时间标注。

原始数据位于 `data/raw/aishell4/test/`，受 `.gitignore` 管理，不进入代码仓库。

## 接入改造

项目新增了：

- 通用二维阵列几何和 8 通道均匀圆阵；
- 支持按时间寻址的多通道 WAV/FLAC 加载器；
- TextGrid 解析和无重叠语音片段筛选；
- SRP-PHAT 宽带粗定位；
- 圆阵宽带 SBL-INCM MVDR；
- Whisper-small 中文 ASR 和繁简统一 CER；
- 固定扇区、自适应扇区及置信度门控消融。

AISHELL-4 没有视频，因此这里用 SRP-PHAT 产生粗方向及置信度。MISP 数据到位后，
将把这个声学先验替换为人脸/唇动视觉先验，SBL-INCM 后端无需重写。

## 复现

检查语料：

```powershell
python scripts/inspect_aishell4.py --root data/raw/aishell4/test
```

运行一条带标注的真实片段：

```powershell
python scripts/evaluate_aishell4_segment.py --enable-asr
```

运行跨 session 小规模评测：

```powershell
# 5场开发子集，用于确定置信度阈值0.20
python scripts/evaluate_aishell4_batch.py --segments 5

# 剩余15场留出子集
python scripts/evaluate_aishell4_batch.py --segments 15 --session-offset 5 `
  --output artifacts/aishell4_batch/heldout_15_metrics.json
```

如调整文本标准化，可直接重算 CER，无需重新运行 Whisper：

```powershell
python scripts/recompute_cer_report.py `
  artifacts/aishell4_batch/heldout_15_metrics.json
```

## 留出15场结果

每场会议自动选择第一个时长3–7秒、具有有效文本且无其他说话人重叠的片段。Whisper
输出先删除标注标签、去除标点并统一为简体中文，再计算 CER。

| 方法 | 平均 CER |
|---|---:|
| 单通道 channel 0 | 43.67% |
| Delay-and-Sum | 38.88% |
| SBL-INCM，固定5°目标扇区 | 41.71% |
| SBL-INCM，置信度自适应扇区 | 38.94% |
| 置信度门控：低置信度 DS，否则自适应 SBL | **38.78%** |

置信度门控方法相对单通道的 CER 降低 11.20%。完整逐句结果见
`artifacts/aishell4_batch/heldout_15_metrics.json`。

## 结果解读和限制

- 固定5°扇区在声学粗方向不准时会把目标能量写入 INCM，真实数据上不够稳健；
- 扩大低置信度目标保护区能降低目标自消除；
- 低置信度时回退到低风险的 Delay-and-Sum，留出集上取得最小平均 CER；
- 当前只评测每场一个无重叠片段，共15条，不是完整 AISHELL-4 官方 benchmark；
- 圆阵半径按公开文献设置为5 cm，需要补充4/5/10 cm 阵列失配敏感性实验；
- 当前使用声学先验，不能把这组结果描述为音视频融合结果；
- 后续应覆盖更多片段、重叠说话、不同房间和置信度阈值，并报告置信区间。
