from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cabin_speech.annotations import (
    SpeechInterval,
    read_textgrid_intervals,
    select_isolated_speech,
)
from cabin_speech.asr import WhisperASR
from cabin_speech.audio import delay_and_sum
from cabin_speech.datasets import (
    MultichannelRecording,
    discover_aishell4_recordings,
    load_multichannel_audio,
)
from cabin_speech.geometry import CircularArrayGeometry
from cabin_speech.localization import DOAResult, estimate_srp_phat_doa
from cabin_speech.metrics import character_error_rate, normalize_chinese_text
from cabin_speech.realtime import SpeechSegment
from cabin_speech.sbl_mvdr import SBLConfig, WidebandConfig, wideband_sbl_mvdr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small real-data ablation on isolated AISHELL-4 speech intervals."
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw/aishell4/test"))
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--session-offset", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--array-radius-m", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/aishell4_batch/metrics.json"),
    )
    return parser.parse_args()


def select_across_sessions(
    root: Path,
    count: int,
    session_offset: int,
) -> list[tuple[Path, SpeechInterval]]:
    selected: list[tuple[Path, SpeechInterval]] = []
    eligible_session_index = 0
    for recording_path in discover_aishell4_recordings(root):
        textgrid_path = root / "TextGrid" / f"{recording_path.stem}.TextGrid"
        if not textgrid_path.exists():
            continue
        intervals = select_isolated_speech(
            read_textgrid_intervals(textgrid_path),
            min_duration_seconds=3.0,
            max_duration_seconds=7.0,
        )
        if intervals:
            if eligible_session_index >= session_offset:
                selected.append((recording_path, intervals[0]))
            eligible_session_index += 1
        if len(selected) >= count:
            break
    return selected


def transcribe(
    asr: WhisperASR,
    audio: np.ndarray,
    sample_rate: int,
    reference: str,
) -> dict[str, float | str]:
    result = asr.transcribe(SpeechSegment(np.asarray(audio, dtype=np.float32), sample_rate))
    return {
        "text": result.text,
        "cer": character_error_rate(
            normalize_chinese_text(reference),
            normalize_chinese_text(result.text),
        ),
        "latency_seconds": result.latency_seconds,
        "real_time_factor": result.real_time_factor,
    }


def enhance(
    recording: MultichannelRecording,
    geometry: CircularArrayGeometry,
    doa: DOAResult,
    confidence_adaptive: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    result = wideband_sbl_mvdr(
        recording.audio,
        recording.sample_rate,
        geometry,
        visual_target_angle_deg=doa.angle_deg,
        visual_confidence=doa.confidence if confidence_adaptive else None,
        sbl_config=SBLConfig(max_iterations=40),
        wideband_config=WidebandConfig(
            n_fft=256,
            hop_length=128,
            frequency_stride=2,
            angle_min_deg=-180.0,
            angle_max_deg=175.0,
            angle_step_deg=5.0,
        ),
    )
    elapsed = time.perf_counter() - started
    return result.enhanced_audio, {
        "sector_half_width_deg": result.sector_half_width_deg,
        "median_target_angle_deg": float(np.median(result.target_angles_deg)),
        "runtime_seconds": elapsed,
        "real_time_factor": elapsed / recording.duration_seconds,
    }


def main() -> None:
    args = parse_args()
    if args.segments < 1:
        raise ValueError("segments must be positive")
    chosen = select_across_sessions(
        args.root,
        args.segments,
        args.session_offset,
    )
    if len(chosen) < args.segments:
        raise RuntimeError(f"only found {len(chosen)} eligible sessions")

    geometry = CircularArrayGeometry(radius_m=args.array_radius_m)
    asr = WhisperASR()
    rows: list[dict[str, object]] = []
    for recording_path, interval in chosen:
        recording = load_multichannel_audio(
            recording_path,
            expected_channels=8,
            start_seconds=interval.start_seconds,
            duration_seconds=interval.duration_seconds,
        )
        doa = estimate_srp_phat_doa(
            recording.audio,
            recording.sample_rate,
            geometry,
        )
        delay_sum_audio = delay_and_sum(
            recording.audio,
            doa.angle_deg,
            geometry,
            recording.sample_rate,
        )
        fixed_audio, fixed_diagnostics = enhance(
            recording,
            geometry,
            doa,
            confidence_adaptive=False,
        )
        adaptive_audio, adaptive_diagnostics = enhance(
            recording,
            geometry,
            doa,
            confidence_adaptive=True,
        )
        row: dict[str, object] = {
            "recording": recording_path.name,
            "speaker_id": interval.speaker_id,
            "start_seconds": interval.start_seconds,
            "duration_seconds": interval.duration_seconds,
            "reference_text": interval.text,
            "srp_phat_doa_deg": doa.angle_deg,
            "srp_phat_confidence": doa.confidence,
            "fixed_diagnostics": fixed_diagnostics,
            "adaptive_diagnostics": adaptive_diagnostics,
            "asr": {
                "input_channel_0": transcribe(
                    asr,
                    recording.audio[0],
                    recording.sample_rate,
                    interval.text,
                ),
                "delay_sum": transcribe(
                    asr,
                    delay_sum_audio,
                    recording.sample_rate,
                    interval.text,
                ),
                "sbl_fixed_sector": transcribe(
                    asr,
                    fixed_audio,
                    recording.sample_rate,
                    interval.text,
                ),
                "sbl_adaptive_sector": transcribe(
                    asr,
                    adaptive_audio,
                    recording.sample_rate,
                    interval.text,
                ),
            },
        }
        row["confidence_gated_method"] = (
            "sbl_adaptive_sector" if doa.confidence >= args.confidence_threshold else "delay_sum"
        )
        rows.append(row)

    method_names = [
        "input_channel_0",
        "delay_sum",
        "sbl_fixed_sector",
        "sbl_adaptive_sector",
    ]
    mean_cer = {
        method: float(
            np.mean(
                [
                    float(row["asr"][method]["cer"])  # type: ignore[index]
                    for row in rows
                ]
            )
        )
        for method in method_names
    }
    gated_cer_values = [
        float(
            row["asr"][row["confidence_gated_method"]]["cer"]  # type: ignore[index]
        )
        for row in rows
    ]
    mean_cer["confidence_gated"] = float(np.mean(gated_cer_values))
    summary = {
        "dataset": "AISHELL-4 test exploratory subset",
        "segments": len(rows),
        "session_offset": args.session_offset,
        "selection": "first 3-7 second isolated lexical interval from each session",
        "array_radius_m": args.array_radius_m,
        "confidence_threshold": args.confidence_threshold,
        "mean_cer": mean_cer,
        "relative_cer_reduction_vs_input": (
            (mean_cer["input_channel_0"] - mean_cer["confidence_gated"])
            / max(mean_cer["input_channel_0"], np.finfo(float).eps)
        ),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
