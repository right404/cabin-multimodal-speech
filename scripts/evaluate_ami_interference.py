from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

from cabin_speech.annotations import read_ami_participants
from cabin_speech.audio import delay_and_sum
from cabin_speech.config import load_config
from cabin_speech.datasets import load_synchronised_mono_files
from cabin_speech.experiment_io import (
    read_active_speaker_prior,
    require_directory,
    require_file,
    resolve_project_path,
)
from cabin_speech.geometry import CircularArrayGeometry
from cabin_speech.localization import estimate_srp_phat_doa
from cabin_speech.metrics import scale_invariant_sdr
from cabin_speech.sbl_mvdr import SBLConfig, WidebandConfig, wideband_sbl_mvdr

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def aligned_si_sdr(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float:
    sample_count = min(reference.size, estimate.size)
    target = reference[:sample_count] - np.mean(reference[:sample_count])
    prediction = estimate[:sample_count] - np.mean(estimate[:sample_count])
    lags = signal.correlation_lags(sample_count, sample_count, mode="full")
    correlation = signal.correlate(prediction, target, mode="full", method="fft")
    allowed = np.abs(lags) <= round(0.08 * sample_rate)
    lag = int(lags[allowed][np.argmax(np.abs(correlation[allowed]))])
    if lag > 0:
        prediction = prediction[lag:]
        target = target[: prediction.size]
    elif lag < 0:
        target = target[-lag:]
        prediction = prediction[: target.size]
    return scale_invariant_sdr(target, prediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMI real-room directional-interference test")
    parser.add_argument("--config", type=Path, default=Path("configs/ami_es2002a.yaml"))
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--active-speaker-results", type=Path)
    parser.add_argument("--interference-db", type=float)
    parser.add_argument("--sbl-max-iterations", type=int)
    parser.add_argument("--frequency-stride", type=int)
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
    experiment = config["interference_experiment"]
    meeting = str(config["meeting_id"])
    meeting_root = require_directory(
        resolve_project_path(PROJECT_ROOT, path_config["ami_root"]) / meeting,
        f"AMI meeting {meeting}",
    )
    audio_root = require_directory(
        args.audio_root or meeting_root / "audio",
        "AMI meeting audio directory",
    )
    annotation_root = require_directory(
        configured_path(args.annotation_root, path_config["annotation_root"]),
        "AMI manual annotation root",
    )
    active_results_path = require_file(
        configured_path(
            args.active_speaker_results,
            path_config["active_speaker_output"],
        ),
        "active-speaker result JSON",
    )
    output_path = configured_path(args.output, path_config["interference_output"])

    target_spec = experiment["target"]
    interferer_spec = experiment["interferer"]
    target_start = float(target_spec["start_seconds"])
    interferer_start = float(interferer_spec["start_seconds"])
    duration = float(experiment["duration_seconds"])
    interference_db = (
        args.interference_db
        if args.interference_db is not None
        else float(experiment["interference_db"])
    )
    sbl_iterations = (
        args.sbl_max_iterations
        if args.sbl_max_iterations is not None
        else int(experiment["sbl_max_iterations"])
    )
    frequency_stride = (
        args.frequency_stride
        if args.frequency_stride is not None
        else int(experiment["frequency_stride"])
    )
    if duration <= 0:
        raise ValueError("experiment duration must be positive")
    if sbl_iterations < 1 or frequency_stride < 1:
        raise ValueError("SBL iterations and frequency stride must be positive")

    visual_prior = read_active_speaker_prior(active_results_path, target_start)
    visual_speaker = str(visual_prior["prediction"])
    speaker_angles = {
        str(speaker): float(angle) for speaker, angle in config["speaker_angles_deg"].items()
    }
    if visual_speaker not in speaker_angles:
        raise ValueError(f"no calibrated direction for predicted speaker {visual_speaker!r}")
    visual_target_angle = speaker_angles[visual_speaker]
    target_speaker = str(target_spec["speaker_id"])
    interferer_speaker = str(interferer_spec["speaker_id"])
    if str(visual_prior["speaker"]) != target_speaker:
        raise ValueError(
            "configured target speaker does not match the active-speaker result ground truth"
        )

    meetings_xml = require_file(
        annotation_root / "corpusResources" / "meetings.xml",
        "AMI meetings.xml",
    )
    participants = read_ami_participants(meetings_xml, meeting)
    headset_channel = participants[target_speaker].headset_channel
    array_config = config["array"]
    array_name = str(array_config["name"])
    microphone_count = int(array_config["microphone_count"])
    array_paths = [
        require_file(
            audio_root / f"{meeting}.{array_name}-{channel:02d}.wav",
            f"AMI array channel {channel}",
        )
        for channel in range(1, microphone_count + 1)
    ]
    reference_path = require_file(
        audio_root / f"{meeting}.Headset-{headset_channel}.wav",
        f"AMI headset channel {headset_channel}",
    )

    target = load_synchronised_mono_files(array_paths, target_start, duration)
    interferer = load_synchronised_mono_files(
        array_paths,
        interferer_start,
        duration,
    )
    reference = load_synchronised_mono_files(
        [reference_path],
        target_start,
        duration,
    ).audio[0]
    scale = np.sqrt(np.mean(target.audio**2) / max(np.mean(interferer.audio**2), 1e-12)) * 10 ** (
        interference_db / 20.0
    )
    mixture = target.audio + scale * interferer.audio

    geometry = CircularArrayGeometry(
        microphone_count=microphone_count,
        radius_m=float(array_config["radius_m"]),
    )
    acoustic_doa = estimate_srp_phat_doa(
        mixture,
        target.sample_rate,
        geometry,
        n_fft=512,
        hop_length=256,
        max_frequency_hz=3_500,
    )
    visual_ds = delay_and_sum(
        mixture,
        visual_target_angle,
        geometry,
        target.sample_rate,
    )
    acoustic_ds = delay_and_sum(
        mixture,
        acoustic_doa.angle_deg,
        geometry,
        target.sample_rate,
    )
    light_asd_probability = float(visual_prior["mean_probability"][visual_speaker])
    beamforming_confidence = float(experiment["beamforming_confidence"])
    sbl_result = wideband_sbl_mvdr(
        mixture,
        target.sample_rate,
        geometry,
        visual_target_angle,
        visual_confidence=beamforming_confidence,
        sbl_config=SBLConfig(max_iterations=sbl_iterations, tolerance=1e-4),
        wideband_config=WidebandConfig(
            n_fft=512,
            hop_length=128,
            min_frequency_hz=300,
            max_frequency_hz=3_500,
            frequency_stride=frequency_stride,
            angle_min_deg=-180,
            angle_max_deg=176,
            angle_step_deg=4,
            default_sector_half_width_deg=8,
            min_sector_half_width_deg=4,
            max_sector_half_width_deg=15,
        ),
    )

    waveforms = {
        "single_channel": mixture[0],
        "audio_doa_delay_sum": acoustic_ds,
        "visual_delay_sum": visual_ds,
        "visual_sbl_incm_mvdr": sbl_result.enhanced_audio,
    }
    scores = {
        name: aligned_si_sdr(reference, waveform, target.sample_rate)
        for name, waveform in waveforms.items()
    }
    output = {
        "schema_version": 1,
        "meeting": meeting,
        "inputs": {
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "active_speaker_results": str(active_results_path.relative_to(PROJECT_ROOT)),
            "array_channels": microphone_count,
            "reference_headset_channel": headset_channel,
        },
        "construction": {
            "target": (
                f"speaker {target_speaker}, {target_start:.3f}-{target_start + duration:.3f} s"
            ),
            "interferer": (
                f"speaker {interferer_speaker}, {interferer_start:.3f}-"
                f"{interferer_start + duration:.3f} s"
            ),
            "interference_db_relative_to_target": interference_db,
            "mixture_type": "controlled sum of two real multichannel room recordings",
        },
        "visual_prior": {
            "source": "persisted Light-ASD active-speaker result",
            "segment_start_seconds": target_start,
            "ground_truth_speaker": target_speaker,
            "predicted_speaker": visual_speaker,
            "target_visible": bool(visual_prior["target_visible"]),
            "predicted_probability": light_asd_probability,
            "beamforming_confidence": beamforming_confidence,
            "confidence_note": (
                "controlled pilot setting; not calibrated from the Light-ASD probability"
            ),
            "identity_to_angle_mapping": "ES2002a-specific empirical calibration",
        },
        "directions_deg": {
            "visual_target": visual_target_angle,
            "interferer": speaker_angles[interferer_speaker],
            "acoustic_estimate": acoustic_doa.angle_deg,
            "acoustic_confidence": acoustic_doa.confidence,
            "sbl_target_median": float(np.median(sbl_result.target_angles_deg)),
        },
        "aligned_si_sdr_db": scores,
        "delta_vs_audio_doa_db": {
            name: value - scores["audio_doa_delay_sum"] for name, value in scores.items()
        },
        "sbl": {
            "max_iterations": sbl_iterations,
            "frequency_stride": frequency_stride,
            "mean_iterations": sbl_result.mean_sbl_iterations,
            "converged_fraction": sbl_result.sbl_converged_fraction,
            "processed_bin_fraction": sbl_result.processed_bin_fraction,
            "protected_sector_half_width_deg": sbl_result.sector_half_width_deg,
            "unprocessed_bin_fallback": "visual delay-and-sum",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
