from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify internal consistency of saved AMI results")
    parser.add_argument(
        "--active-speaker",
        type=Path,
        default=Path("artifacts/ami/active_speaker_results.json"),
    )
    parser.add_argument(
        "--interference",
        type=Path,
        default=Path("artifacts/ami/interference_results.json"),
    )
    return parser.parse_args()


def require_close(actual: float, expected: float, name: str, tolerance: float = 1e-10) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{name} mismatch: saved={actual}, recomputed={expected}")


def main() -> None:
    args = parse_args()
    active = json.loads(args.active_speaker.read_text(encoding="utf-8"))
    interference = json.loads(args.interference.read_text(encoding="utf-8"))

    segments = active["segments"]
    visible = [segment for segment in segments if segment["target_visible"]]
    recomputed = {
        "all_correct": sum(bool(segment["correct"]) for segment in segments),
        "all_total": len(segments),
        "visible_correct": sum(bool(segment["correct"]) for segment in visible),
        "visible_total": len(visible),
    }
    summary = active["summary"]
    if summary["all_segments"]["correct"] != recomputed["all_correct"]:
        raise ValueError("active-speaker all-segment correct count is inconsistent")
    if summary["all_segments"]["total"] != recomputed["all_total"]:
        raise ValueError("active-speaker all-segment total is inconsistent")
    if summary["visible_segments"]["correct"] != recomputed["visible_correct"]:
        raise ValueError("active-speaker visible correct count is inconsistent")
    if summary["visible_segments"]["total"] != recomputed["visible_total"]:
        raise ValueError("active-speaker visible total is inconsistent")
    require_close(
        float(summary["all_segments"]["accuracy"]),
        recomputed["all_correct"] / recomputed["all_total"],
        "all-segment accuracy",
    )
    require_close(
        float(summary["visible_segments"]["accuracy"]),
        recomputed["visible_correct"] / recomputed["visible_total"],
        "visible-segment accuracy",
    )
    require_close(
        float(summary["visual_coverage"]),
        recomputed["visible_total"] / recomputed["all_total"],
        "visual coverage",
    )

    prior = interference["visual_prior"]
    target_start = float(prior["segment_start_seconds"])
    target_segments = [
        segment for segment in segments if abs(float(segment["start"]) - target_start) < 1e-3
    ]
    if len(target_segments) != 1:
        raise ValueError("interference visual-prior source segment is missing or ambiguous")
    target_segment = target_segments[0]
    if target_segment["prediction"] != prior["predicted_speaker"]:
        raise ValueError("interference predicted speaker differs from active-speaker JSON")
    require_close(
        float(target_segment["mean_probability"][target_segment["prediction"]]),
        float(prior["predicted_probability"]),
        "persisted Light-ASD probability",
    )

    scores = interference["aligned_si_sdr_db"]
    deltas = interference["delta_vs_audio_doa_db"]
    baseline = float(scores["audio_doa_delay_sum"])
    for method, score in scores.items():
        require_close(
            float(deltas[method]),
            float(score) - baseline,
            f"{method} SI-SDR delta",
        )

    verified = {
        "active_speaker": {
            "all_segments": (f"{recomputed['all_correct']}/{recomputed['all_total']}"),
            "visible_segments": (f"{recomputed['visible_correct']}/{recomputed['visible_total']}"),
            "visual_coverage": summary["visual_coverage"],
        },
        "interference": {
            "predicted_speaker": prior["predicted_speaker"],
            "visual_target_deg": interference["directions_deg"]["visual_target"],
            "acoustic_target_deg": interference["directions_deg"]["acoustic_estimate"],
            "visual_ds_delta_db": deltas["visual_delay_sum"],
            "visual_sbl_delta_db": deltas["visual_sbl_incm_mvdr"],
        },
    }
    print(json.dumps(verified, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
