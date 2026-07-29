from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_fusion import PROJECT_ROOT, build_datasets

from cabin_speech.config import load_config
from cabin_speech.fusion import MultimodalFusionNet
from cabin_speech.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate modality ablations.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="artifacts/baseline/best_model.pt")
    parser.add_argument("--output", default="artifacts/baseline/ablation_metrics.json")
    return parser.parse_args()


def mask_features(features: torch.Tensor, mode: str) -> torch.Tensor:
    masked = features.clone()
    if mode == "audio_only":
        masked[:, 4:12] = 0.0
        masked[:, 13] = 0.0
    elif mode == "visual_only":
        masked[:, 0:4] = 0.0
        masked[:, 12] = 0.0
    elif mode != "audio_visual":
        raise ValueError(f"unknown ablation mode: {mode}")
    return masked


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    _, _, test_data = build_datasets(config)
    checkpoint = torch.load(PROJECT_ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = MultimodalFusionNet(
        checkpoint["input_dim"],
        checkpoint["num_classes"],
        checkpoint["hidden_dims"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(test_data, batch_size=512, shuffle=False)
    targets: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {
        "audio_only": [],
        "visual_only": [],
        "audio_visual": [],
    }
    with torch.inference_mode():
        for features, labels in loader:
            targets.append(labels.numpy())
            for mode in predictions:
                logits = model(mask_features(features, mode))
                predictions[mode].append(logits.argmax(dim=1).numpy())

    target_array = np.concatenate(targets)
    results = {
        mode: classification_metrics(
            target_array,
            np.concatenate(mode_predictions),
            checkpoint["class_names"],
        )
        for mode, mode_predictions in predictions.items()
    }
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, metrics in results.items():
        print(f"{mode:>12} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
