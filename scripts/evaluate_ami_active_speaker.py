from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from cabin_speech.active_speaker import LightASDInference, crop_face_sequence
from cabin_speech.annotations import (
    read_ami_participants,
    read_ami_speech_intervals,
    select_isolated_speech,
)
from cabin_speech.config import load_config
from cabin_speech.experiment_io import (
    require_directory,
    require_file,
    resolve_project_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate active-speaker ID on AMI close-ups")
    parser.add_argument("--config", type=Path, default=Path("configs/ami_es2002a.yaml"))
    parser.add_argument("--ami-root", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--meeting")
    parser.add_argument("--light-asd-root", type=Path)
    parser.add_argument("--yunet-model", type=Path)
    parser.add_argument(
        "--weights",
        choices=["pretrain_AVA_CVPR.model", "finetuning_TalkSet.model"],
    )
    parser.add_argument("--visibility-threshold", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def configured_path(
    explicit: Path | None,
    config_value: str,
) -> Path:
    return explicit if explicit is not None else resolve_project_path(PROJECT_ROOT, config_value)


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_config(config_path)
    path_config = config["paths"]
    active_config = config["active_speaker"]

    meeting = args.meeting or str(config["meeting_id"])
    ami_root = require_directory(
        configured_path(args.ami_root, path_config["ami_root"]),
        "AMI corpus root",
    )
    annotations = require_directory(
        configured_path(args.annotation_root, path_config["annotation_root"]),
        "AMI manual annotation root",
    )
    light_asd_root = require_directory(
        configured_path(args.light_asd_root, path_config["light_asd_root"]),
        "Light-ASD repository",
    )
    yunet_model = require_file(
        configured_path(args.yunet_model, path_config["yunet_model"]),
        "YuNet ONNX model",
    )
    weights_name = args.weights or str(active_config["weights"])
    weights_path = require_file(
        light_asd_root / "weight" / weights_name,
        "Light-ASD weights",
    )
    output_path = configured_path(args.output, path_config["active_speaker_output"])
    visibility_threshold = (
        args.visibility_threshold
        if args.visibility_threshold is not None
        else float(active_config["visibility_threshold"])
    )
    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("visibility-threshold must be between 0 and 1")
    seed = args.seed if args.seed is not None else int(active_config["seed"])
    np.random.seed(seed)

    meeting_root = require_directory(ami_root / meeting, f"AMI meeting {meeting}")
    meetings_xml = require_file(
        annotations / "corpusResources" / "meetings.xml",
        "AMI meetings.xml",
    )
    participants = read_ami_participants(meetings_xml, meeting)
    intervals = select_isolated_speech(
        read_ami_speech_intervals(annotations, meeting),
        min_duration_seconds=float(active_config["min_segment_seconds"]),
        max_duration_seconds=float(active_config["max_segment_seconds"]),
    )
    if not intervals:
        raise RuntimeError(f"no eligible isolated speech intervals found for {meeting}")

    array_name = str(config["array"]["name"])
    array_channel = require_file(
        meeting_root / "audio" / f"{meeting}.{array_name}-01.wav",
        "AMI array channel 1",
    )
    for participant in participants.values():
        require_file(
            meeting_root / "video" / f"{meeting}.{participant.camera_id}.avi",
            f"AMI video {participant.camera_id}",
        )

    model = LightASDInference(
        light_asd_root,
        weights_path,
        device=args.device,
    )
    segments = []
    started = time.perf_counter()
    with sf.SoundFile(array_channel) as audio_file:
        sample_rate = int(audio_file.samplerate)
        for index, interval in enumerate(intervals, start=1):
            audio_file.seek(round(interval.start_seconds * sample_rate))
            audio = audio_file.read(
                round(interval.duration_seconds * sample_rate),
                dtype="float32",
            )

            probabilities: dict[str, float] = {}
            detection_rates: dict[str, float] = {}
            for agent_id, participant in participants.items():
                video_path = meeting_root / "video" / f"{meeting}.{participant.camera_id}.avi"
                frames, detection_rate = crop_face_sequence(
                    video_path,
                    interval.start_seconds,
                    interval.end_seconds,
                    yunet_model,
                )
                probabilities[agent_id] = float(np.mean(model.predict(audio, sample_rate, frames)))
                detection_rates[agent_id] = detection_rate

            prediction = max(probabilities, key=probabilities.get)
            target_visible = detection_rates[interval.speaker_id] >= visibility_threshold
            segments.append(
                {
                    "start": interval.start_seconds,
                    "end": interval.end_seconds,
                    "speaker": interval.speaker_id,
                    "text": interval.text,
                    "prediction": prediction,
                    "correct": prediction == interval.speaker_id,
                    "target_visible": target_visible,
                    "mean_probability": probabilities,
                    "face_detection_rate": detection_rates,
                }
            )
            print(
                f"{index:02d}/{len(intervals)} true={interval.speaker_id} "
                f"prediction={prediction} visible={target_visible}"
            )

    visible = [segment for segment in segments if segment["target_visible"]]
    output = {
        "schema_version": 1,
        "meeting": meeting,
        "weights": weights_name,
        "device": str(model.device),
        "seed": seed,
        "selection": "isolated lexical segments, 2-8 seconds",
        "visibility_threshold": visibility_threshold,
        "inputs": {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "array_audio": str(array_channel.relative_to(PROJECT_ROOT)),
            "annotation_root": str(annotations.relative_to(PROJECT_ROOT)),
            "yunet_model": str(yunet_model.relative_to(PROJECT_ROOT)),
            "light_asd_weights": str(weights_path.relative_to(PROJECT_ROOT)),
        },
        "summary": {
            "all_segments": {
                "correct": sum(segment["correct"] for segment in segments),
                "total": len(segments),
                "accuracy": float(np.mean([segment["correct"] for segment in segments])),
            },
            "visible_segments": {
                "correct": sum(segment["correct"] for segment in visible),
                "total": len(visible),
                "accuracy": float(np.mean([segment["correct"] for segment in visible])),
            },
            "visual_coverage": len(visible) / len(segments),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "segments": segments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
