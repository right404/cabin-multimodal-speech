from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from cabin_speech.audio import delay_and_sum, simulate_plane_wave
from cabin_speech.geometry import CabinGeometry
from cabin_speech.metrics import scale_invariant_sdr
from cabin_speech.sbl_mvdr import (
    SBLConfig,
    WidebandConfig,
    wideband_sbl_mvdr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible visual-guided wideband SBL-INCM MVDR simulation."
    )
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--microphones", type=int, default=6)
    parser.add_argument("--microphone-spacing-m", type=float, default=0.04)
    parser.add_argument("--target-angle", type=float, default=20.0)
    parser.add_argument("--visual-angle", type=float, default=23.0)
    parser.add_argument("--visual-confidence", type=float, default=0.88)
    parser.add_argument("--interference-angle", type=float, default=-42.0)
    parser.add_argument("--interference-gain", type=float, default=1.7)
    parser.add_argument(
        "--fixed-sector",
        action="store_true",
        help="Disable the visual-confidence adaptation and use the paper's fixed 5-degree sector.",
    )
    parser.add_argument("--paper-settings", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sbl_mvdr_demo"))
    return parser.parse_args()


def speech_like_signal(
    time_axis: np.ndarray,
    fundamental_hz: float,
    phase: float = 0.0,
) -> np.ndarray:
    envelope = np.square(np.maximum(np.sin(2.0 * np.pi * 1.7 * time_axis + phase), 0.0))
    signal_value = np.zeros_like(time_axis)
    for harmonic, amplitude in ((1, 1.0), (2, 0.55), (3, 0.3), (5, 0.16)):
        instantaneous_phase = 2.0 * np.pi * harmonic * fundamental_hz * time_axis + 0.8 * np.sin(
            2.0 * np.pi * 2.1 * time_axis
        )
        signal_value += amplitude * np.sin(instantaneous_phase + phase)
    return envelope * signal_value / max(np.max(np.abs(signal_value)), 1e-12)


def write_audio(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    peak = max(float(np.max(np.abs(audio))), 1e-12)
    normalized = np.clip(0.95 * audio / peak, -1.0, 1.0)
    wavfile.write(path, sample_rate, np.asarray(normalized * 32767.0, dtype=np.int16))


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be positive")
    rng = np.random.default_rng(args.seed)
    sample_rate = 16_000
    samples = int(round(args.duration * sample_rate))
    time_axis = np.arange(samples, dtype=np.float64) / sample_rate
    geometry = CabinGeometry(
        microphone_count=args.microphones,
        microphone_spacing_m=args.microphone_spacing_m,
    )

    true_target_angle = args.target_angle
    visual_angle = args.visual_angle
    visual_confidence = args.visual_confidence
    interference_angle = args.interference_angle
    target = speech_like_signal(time_axis, fundamental_hz=145.0)
    interference = args.interference_gain * speech_like_signal(
        time_axis,
        fundamental_hz=165.0,
        phase=0.8,
    )
    target_channels = simulate_plane_wave(target, true_target_angle, geometry, sample_rate)
    interference_channels = simulate_plane_wave(
        interference,
        interference_angle,
        geometry,
        sample_rate,
    )
    noise = rng.normal(scale=0.08, size=target_channels.shape)
    mixture = target_channels + interference_channels + noise

    delay_sum = delay_and_sum(mixture, visual_angle, geometry, sample_rate)
    if args.paper_settings:
        sbl_config = SBLConfig(max_iterations=200)
        wideband_config = WidebandConfig(
            angle_step_deg=1.0,
            frequency_stride=1,
        )
    else:
        sbl_config = SBLConfig(max_iterations=40)
        wideband_config = WidebandConfig(
            n_fft=256,
            hop_length=128,
            angle_step_deg=5.0,
            frequency_stride=1,
        )
    if args.fixed_sector:
        wideband_config = replace(
            wideband_config,
            min_sector_half_width_deg=5.0,
            max_sector_half_width_deg=5.0,
        )

    start = time.perf_counter()
    result = wideband_sbl_mvdr(
        mixture,
        sample_rate,
        geometry,
        visual_target_angle_deg=visual_angle,
        visual_confidence=visual_confidence,
        sbl_config=sbl_config,
        wideband_config=wideband_config,
    )
    elapsed_seconds = time.perf_counter() - start

    reference_channel = int(np.argmin(np.abs(geometry.microphone_positions_m)))
    input_audio = mixture[reference_channel]
    metrics = {
        "input_si_sdr_db": scale_invariant_sdr(target, input_audio),
        "delay_sum_si_sdr_db": scale_invariant_sdr(target, delay_sum),
        "sbl_incm_mvdr_si_sdr_db": scale_invariant_sdr(target, result.enhanced_audio),
        "improvement_over_input_db": (
            scale_invariant_sdr(target, result.enhanced_audio)
            - scale_invariant_sdr(target, input_audio)
        ),
        "runtime_seconds": elapsed_seconds,
        "audio_seconds": args.duration,
        "real_time_factor": elapsed_seconds / args.duration,
        "mean_sbl_iterations": result.mean_sbl_iterations,
        "median_estimated_target_angle_deg": float(np.median(result.target_angles_deg)),
        "visual_target_angle_deg": visual_angle,
        "true_target_angle_deg": true_target_angle,
        "interference_angle_deg": interference_angle,
        "visual_confidence": visual_confidence,
        "protected_sector_half_width_deg": result.sector_half_width_deg,
        "paper_settings": bool(args.paper_settings),
        "fixed_sector": bool(args.fixed_sector),
        "microphones": args.microphones,
        "microphone_spacing_m": args.microphone_spacing_m,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_audio(args.output_dir / "mixture.wav", sample_rate, input_audio)
    write_audio(args.output_dir / "delay_sum.wav", sample_rate, delay_sum)
    write_audio(args.output_dir / "sbl_incm_mvdr.wav", sample_rate, result.enhanced_audio)
    write_audio(args.output_dir / "reference.wav", sample_rate, target)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
