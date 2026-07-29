from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cabin_speech.config import load_config
from cabin_speech.fusion import MultimodalFusionNet
from cabin_speech.geometry import CabinGeometry
from cabin_speech.metrics import classification_metrics
from cabin_speech.synthetic import SyntheticCabinDataset, SyntheticDataConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the cabin audio-visual fusion baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output-dir", default="artifacts/baseline")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_datasets(
    config: dict[str, Any],
) -> tuple[SyntheticCabinDataset, SyntheticCabinDataset, SyntheticCabinDataset]:
    geometry_config = config["geometry"]
    geometry = CabinGeometry(
        seat_angles_deg=tuple(geometry_config["seat_angles_deg"]),
        seat_names=tuple(geometry_config["seat_names"]),
        microphone_count=int(geometry_config["microphone_count"]),
        microphone_spacing_m=float(geometry_config["microphone_spacing_m"]),
        speed_of_sound_m_s=float(geometry_config["speed_of_sound_m_s"]),
    )
    data_config = config["data"]
    synthetic_config = SyntheticDataConfig(
        no_speech_probability=float(data_config["no_speech_probability"]),
        audio_missing_probability=float(data_config["audio_missing_probability"]),
        visual_occlusion_probability=float(data_config["visual_occlusion_probability"]),
        distractor_lip_probability=float(data_config["distractor_lip_probability"]),
        doa_noise_std_deg=float(data_config["doa_noise_std_deg"]),
    )
    seed = int(config["seed"])
    train = SyntheticCabinDataset(
        int(data_config["train_samples"]), geometry, synthetic_config, seed
    )
    validation = SyntheticCabinDataset(
        int(data_config["validation_samples"]), geometry, synthetic_config, seed + 1
    )
    test = SyntheticCabinDataset(
        int(data_config["test_samples"]), geometry, synthetic_config, seed + 2
    )
    return train, validation, test


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * labels.size(0)
        targets.append(labels.detach().cpu().numpy())
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    return average_loss, np.concatenate(targets), np.concatenate(predictions)


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["seed"]))
    train_data, validation_data, test_data = build_datasets(config)
    training_config = config["training"]
    device = select_device(str(training_config["device"]))
    print(
        f"device={device} torch={torch.__version__} cuda_runtime={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}"
    )

    loaders = {
        "train": DataLoader(
            train_data,
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            num_workers=int(training_config["num_workers"]),
            pin_memory=device.type == "cuda",
        ),
        "validation": DataLoader(
            validation_data,
            batch_size=int(training_config["batch_size"]),
            shuffle=False,
            num_workers=int(training_config["num_workers"]),
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            test_data,
            batch_size=int(training_config["batch_size"]),
            shuffle=False,
            num_workers=int(training_config["num_workers"]),
            pin_memory=device.type == "cuda",
        ),
    }

    model_config = config["model"]
    model = MultimodalFusionNet(
        input_dim=train_data.feature_dim,
        num_classes=train_data.num_classes,
        hidden_dims=tuple(model_config["hidden_dims"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    class_names = [*train_data.geometry.seat_names, "no_speech"]
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    model_path = output_dir / "best_model.pt"

    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_loss, train_targets, train_predictions = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        validation_loss, validation_targets, validation_predictions = run_epoch(
            model, loaders["validation"], criterion, device
        )
        train_metrics = classification_metrics(train_targets, train_predictions, class_names)
        validation_metrics = classification_metrics(
            validation_targets, validation_predictions, class_names
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}"
        )
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = float(validation_metrics["macro_f1"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": train_data.feature_dim,
                    "num_classes": train_data.num_classes,
                    "hidden_dims": list(model_config["hidden_dims"]),
                    "dropout": float(model_config["dropout"]),
                    "class_names": class_names,
                    "config": config,
                },
                model_path,
            )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_targets, test_predictions = run_epoch(model, loaders["test"], criterion, device)
    test_metrics = classification_metrics(test_targets, test_predictions, class_names)
    test_metrics["loss"] = test_loss
    test_metrics["device"] = str(device)
    test_metrics["torch_version"] = torch.__version__
    test_metrics["cuda_runtime"] = torch.version.cuda

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"test_loss={test_loss:.4f} test_acc={test_metrics['accuracy']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"saved={model_path}")


if __name__ == "__main__":
    main()
