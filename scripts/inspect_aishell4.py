from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cabin_speech.datasets import (
    discover_aishell4_recordings,
    load_multichannel_audio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an extracted AISHELL-4 corpus.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/aishell4/test"),
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recordings = discover_aishell4_recordings(args.root)
    if not recordings:
        raise FileNotFoundError(f"no WAV recordings found below {args.root}")
    recording = load_multichannel_audio(
        recordings[0],
        expected_channels=8,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
    )
    channel_rms = np.sqrt(np.mean(np.square(recording.audio), axis=1))
    result = {
        "corpus_root": str(args.root.resolve()),
        "recording_count": len(recordings),
        "first_recording": str(recording.source_path),
        "sample_rate": recording.sample_rate,
        "channels": int(recording.audio.shape[0]),
        "loaded_samples": int(recording.audio.shape[1]),
        "loaded_duration_seconds": recording.duration_seconds,
        "channel_rms": channel_rms.tolist(),
        "finite": bool(np.isfinite(recording.audio).all()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
