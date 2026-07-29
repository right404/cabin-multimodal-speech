# 第三方数据、模型与代码

仓库只保存项目源码、配置和小型 JSON 结果，不保存以下大型数据、模型权重或第三方仓库。使用者应自行阅读并遵守对应许可。

| 资源 | 项目用途 | 来源 | 许可/说明 |
|---|---|---|---|
| AMI Meeting Corpus | `ES2002a` 同步阵列、视频、头戴麦克风和标注 | <https://groups.inf.ed.ac.uk/ami/corpus/> | AMI 官方页面标注为 CC BY 4.0；使用时保留署名 |
| Light-ASD | 主动说话人推理代码和预训练权重 | <https://github.com/Junhua-Liao/Light-ASD> | 上游仓库 MIT License |
| OpenCV Zoo YuNet | 人脸检测 ONNX | <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet> | 以模型目录和上游仓库许可为准 |
| AISHELL-4 / SLR111 | 音频-only 8 通道补充实验 | <https://www.openslr.org/111/> | CC BY-SA 4.0 |
| OpenAI Whisper | 实时 Demo 可选 ASR | <https://github.com/openai/whisper> | 上游仓库 MIT License；模型下载还应查看对应 model card |
| SBL-INCM-MVDR 论文 | 算法公式和实现依据 | <https://doi.org/10.1016/j.dsp.2026.106138> | 论文 PDF 不随仓库提交 |

本项目通过 `.gitignore` 排除：

- `data/raw/` 和 `data/processed/`；
- `*.onnx`、`*.pt`、`*.pth`；
- 音频、视频和本地论文 PDF；
- 第三方 Light-ASD clone。

提交或发布项目前，应再次检查 `git status`，避免误提交语料、人物视频、模型权重或受版权保护的论文全文。
