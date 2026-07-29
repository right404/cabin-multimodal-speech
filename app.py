from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import torch

from cabin_speech.fusion import MultimodalFusionNet
from cabin_speech.geometry import CabinGeometry

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "baseline" / "best_model.pt"
DISPLAY_NAMES = {
    "driver": "驾驶位",
    "front_passenger": "副驾驶位",
    "rear_left": "左后排",
    "rear_right": "右后排",
    "no_speech": "无人说话",
}


def load_model() -> tuple[MultimodalFusionNet, dict]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "未找到训练权重，请先运行 python scripts/train_fusion.py --config configs/baseline.yaml"
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = MultimodalFusionNet(
        checkpoint["input_dim"],
        checkpoint["num_classes"],
        checkpoint["hidden_dims"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


MODEL, CHECKPOINT = load_model()
GEOMETRY = CabinGeometry()


def predict(
    audio_doa: float,
    audio_confidence: float,
    driver_lip: float,
    passenger_lip: float,
    rear_left_lip: float,
    rear_right_lip: float,
    driver_visibility: float,
    passenger_visibility: float,
    rear_left_visibility: float,
    rear_right_visibility: float,
) -> tuple[dict[str, float], str]:
    audio_confidence = float(np.clip(audio_confidence, 0.0, 1.0))
    audio_scores = GEOMETRY.angle_likelihoods(audio_doa) * audio_confidence
    lip_scores = np.clip(
        np.asarray(
            [driver_lip, passenger_lip, rear_left_lip, rear_right_lip],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )
    visibility = np.clip(
        np.asarray(
            [
                driver_visibility,
                passenger_visibility,
                rear_left_visibility,
                rear_right_visibility,
            ],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )
    visual_confidence = float(visibility.max())
    features = np.concatenate(
        [
            audio_scores,
            lip_scores,
            visibility,
            [audio_confidence, visual_confidence],
        ]
    ).astype(np.float32)
    with torch.inference_mode():
        probabilities = MODEL(torch.from_numpy(features).unsqueeze(0)).softmax(dim=1)[0]

    scores = {
        DISPLAY_NAMES[name]: float(probability)
        for name, probability in zip(CHECKPOINT["class_names"], probabilities.tolist(), strict=True)
    }
    winner = max(scores, key=scores.get)
    summary = (
        f"### 当前判断：{winner}\n\n"
        f"- 融合置信度：{scores[winner]:.1%}\n"
        f"- 音频 DOA：{audio_doa:.1f}°\n"
        f"- 音频置信度：{audio_confidence:.1%}\n\n"
        "> 当前模型使用合成数据训练，界面用于验证融合逻辑；接入真实音视频后再报告业务指标。"
    )
    return scores, summary


with gr.Blocks(title="车载座舱多模态主动说话人感知") as demo:
    gr.Markdown(
        """
        # 车载座舱多模态主动说话人感知
        调节音频声源方向、音频置信度、各座位唇动强度和可见度，
        观察音视频融合模型对活跃说话人座位的判断。
        """
    )
    with gr.Row():
        with gr.Column():
            audio_doa = gr.Slider(-75, 75, value=-55, step=1, label="音频 DOA（度）")
            audio_confidence = gr.Slider(0, 1, value=0.9, step=0.01, label="音频置信度")
        with gr.Column():
            output_label = gr.Label(label="座位概率")
            output_summary = gr.Markdown()

    gr.Markdown("## 视觉唇动与可见度")
    lip_inputs = []
    visibility_inputs = []
    seat_labels = ["驾驶位", "副驾驶位", "左后排", "右后排"]
    default_lips = [0.9, 0.1, 0.05, 0.05]
    with gr.Row():
        for seat_label, default_lip in zip(seat_labels, default_lips, strict=True):
            with gr.Column():
                lip_inputs.append(
                    gr.Slider(
                        0,
                        1,
                        value=default_lip,
                        step=0.01,
                        label=f"{seat_label}唇动",
                    )
                )
                visibility_inputs.append(
                    gr.Slider(
                        0,
                        1,
                        value=1.0,
                        step=0.01,
                        label=f"{seat_label}可见度",
                    )
                )

    run_button = gr.Button("运行多模态融合", variant="primary")
    inputs = [audio_doa, audio_confidence]
    for lip_input, visibility_input in zip(lip_inputs, visibility_inputs, strict=True):
        inputs.extend([lip_input, visibility_input])
    run_button.click(
        fn=lambda doa, confidence, dl, dv, pl, pv, rll, rlv, rrl, rrv: predict(
            doa, confidence, dl, pl, rll, rrl, dv, pv, rlv, rrv
        ),
        inputs=inputs,
        outputs=[output_label, output_summary],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
