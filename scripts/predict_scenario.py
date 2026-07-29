from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cabin_speech.fusion import MultimodalFusionNet
from cabin_speech.geometry import CabinGeometry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one cabin fusion scenario.")
    parser.add_argument("--checkpoint", default="artifacts/baseline/best_model.pt")
    parser.add_argument("--audio-doa", type=float, default=-55.0)
    parser.add_argument("--audio-confidence", type=float, default=0.9)
    parser.add_argument(
        "--lip-scores",
        type=float,
        nargs=4,
        default=[0.9, 0.1, 0.05, 0.05],
        metavar=("DRIVER", "FRONT_PASSENGER", "REAR_LEFT", "REAR_RIGHT"),
    )
    parser.add_argument(
        "--visibility",
        type=float,
        nargs=4,
        default=[1.0, 1.0, 1.0, 1.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MultimodalFusionNet(
        checkpoint["input_dim"],
        checkpoint["num_classes"],
        checkpoint["hidden_dims"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    geometry = CabinGeometry()
    audio_confidence = float(np.clip(args.audio_confidence, 0.0, 1.0))
    audio_scores = geometry.angle_likelihoods(args.audio_doa) * audio_confidence
    lip_scores = np.clip(np.asarray(args.lip_scores, dtype=np.float32), 0.0, 1.0)
    visibility = np.clip(np.asarray(args.visibility, dtype=np.float32), 0.0, 1.0)
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
        probabilities = model(torch.from_numpy(features).unsqueeze(0)).softmax(dim=1)[0]

    class_names = checkpoint["class_names"]
    ranking = sorted(
        zip(class_names, probabilities.tolist(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    for name, probability in ranking:
        print(f"{name:>16}: {probability:.4f}")


if __name__ == "__main__":
    main()
