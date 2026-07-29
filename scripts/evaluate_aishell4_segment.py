from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from cabin_speech.asr import WhisperASR
from cabin_speech.audio import delay_and_sum
from cabin_speech.datasets import load_multichannel_audio
from cabin_speech.geometry import CircularArrayGeometry
from cabin_speech.localization import estimate_srp_phat_doa
from cabin_speech.metrics import character_error_rate, normalize_chinese_text
from cabin_speech.realtime import SpeechSegment
from cabin_speech.sbl_mvdr import (
    INCMConfig,
    SBLConfig,
    WidebandConfig,
    wideband_sbl_mvdr,
)

DEFAULT_RECORDING = Path("data/raw/aishell4/test/wav/L_R003S01C02.flac")
DEFAULT_REFERENCE = "今天啊把各位都叫过来啊主要咱讨论一下咱那个呃咱这个小区"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SRP-PHAT + SBL-INCM MVDR on a real AISHELL-4 segment."
    )
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--start-seconds", type=float, default=23.475)
    parser.add_argument("--duration-seconds", type=float, default=5.01)
    parser.add_argument("--reference-text", default=DEFAULT_REFERENCE)
    parser.add_argument("--array-radius-m", type=float, default=0.05)
    parser.add_argument("--loading-beta", type=float, default=1e-3)
    parser.add_argument(
        "--fixed-sector",
        action="store_true",
        help="Use the paper's fixed 5-degree sector instead of confidence adaptation.",
    )
    parser.add_argument("--enable-asr", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/aishell4_real"))
    return parser.parse_args()


def transcribe(asr: WhisperASR, audio: np.ndarray, sample_rate: int) -> dict[str, float | str]:
    result = asr.transcribe(
        SpeechSegment(
            samples=np.asarray(audio, dtype=np.float32),
            sample_rate=sample_rate,
        )
    )
    return {
        "text": result.text,
        "latency_seconds": result.latency_seconds,
        "real_time_factor": result.real_time_factor,
    }


def main() -> None:
    args = parse_args()
    recording = load_multichannel_audio(
        args.recording,
        expected_channels=8,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
    )
    geometry = CircularArrayGeometry(
        microphone_count=8,
        radius_m=args.array_radius_m,
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

    started = time.perf_counter()
    enhanced = wideband_sbl_mvdr(
        recording.audio,
        recording.sample_rate,
        geometry,
        visual_target_angle_deg=doa.angle_deg,
        visual_confidence=None if args.fixed_sector else doa.confidence,
        sbl_config=SBLConfig(max_iterations=40),
        incm_config=INCMConfig(diagonal_loading_beta=args.loading_beta),
        wideband_config=WidebandConfig(
            n_fft=256,
            hop_length=128,
            min_frequency_hz=200.0,
            max_frequency_hz=6_000.0,
            frequency_stride=2,
            angle_min_deg=-180.0,
            angle_max_deg=175.0,
            angle_step_deg=5.0,
        ),
    )
    enhancement_seconds = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_audio = recording.audio[0]
    sf.write(args.output_dir / "input_channel_0.wav", input_audio, recording.sample_rate)
    sf.write(args.output_dir / "delay_sum.wav", delay_sum_audio, recording.sample_rate)
    sf.write(
        args.output_dir / "sbl_incm_mvdr.wav",
        enhanced.enhanced_audio,
        recording.sample_rate,
    )

    metrics: dict[str, object] = {
        "dataset": "AISHELL-4 test",
        "recording": str(args.recording),
        "start_seconds": args.start_seconds,
        "duration_seconds": recording.duration_seconds,
        "sample_rate": recording.sample_rate,
        "channels": int(recording.audio.shape[0]),
        "array_type": "8-channel uniform circular array",
        "assumed_array_radius_m": args.array_radius_m,
        "prior_source": "audio-only SRP-PHAT (MISP visual data pending)",
        "confidence_adaptive_sector": not args.fixed_sector,
        "protected_sector_half_width_deg": enhanced.sector_half_width_deg,
        "diagonal_loading_beta": args.loading_beta,
        "srp_phat_doa_deg": doa.angle_deg,
        "srp_phat_confidence": doa.confidence,
        "median_sbl_target_angle_deg": float(np.median(enhanced.target_angles_deg)),
        "mean_sbl_iterations": enhanced.mean_sbl_iterations,
        "enhancement_runtime_seconds": enhancement_seconds,
        "enhancement_real_time_factor": (enhancement_seconds / recording.duration_seconds),
        "reference_text": args.reference_text,
    }
    if args.enable_asr:
        asr = WhisperASR()
        asr_results = {
            "input_channel_0": transcribe(asr, input_audio, recording.sample_rate),
            "delay_sum": transcribe(asr, delay_sum_audio, recording.sample_rate),
            "sbl_incm_mvdr": transcribe(
                asr,
                enhanced.enhanced_audio,
                recording.sample_rate,
            ),
        }
        normalized_reference = normalize_chinese_text(args.reference_text)
        for result in asr_results.values():
            result["cer"] = character_error_rate(
                normalized_reference,
                normalize_chinese_text(str(result["text"])),
            )
        metrics["asr"] = asr_results

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
