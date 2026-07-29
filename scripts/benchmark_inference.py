from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from cabin_speech.fusion import MultimodalFusionNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fusion-model inference latency.")
    parser.add_argument("--checkpoint", default="artifacts/baseline/best_model.pt")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    checkpoint = torch.load(PROJECT_ROOT / args.checkpoint, map_location=device, weights_only=False)
    model = MultimodalFusionNet(
        checkpoint["input_dim"],
        checkpoint["num_classes"],
        checkpoint["hidden_dims"],
        checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    features = torch.randn(1, checkpoint["input_dim"], device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(features)
        synchronise(device)

        timings_ms = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            model(features)
            synchronise(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    timings_ms.sort()
    percentile_95 = timings_ms[int(0.95 * (len(timings_ms) - 1))]
    print(
        f"device={device} iterations={args.iterations} "
        f"mean_ms={statistics.fmean(timings_ms):.4f} "
        f"median_ms={statistics.median(timings_ms):.4f} "
        f"p95_ms={percentile_95:.4f}"
    )


if __name__ == "__main__":
    main()
