from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cabin_speech.metrics import character_error_rate, normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute a saved AISHELL-4 report with normalized Chinese CER."
    )
    parser.add_argument("report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report["results"]
    method_names = [
        "input_channel_0",
        "delay_sum",
        "sbl_fixed_sector",
        "sbl_adaptive_sector",
    ]
    for row in rows:
        reference = normalize_chinese_text(row["reference_text"])
        for method in method_names:
            hypothesis = normalize_chinese_text(row["asr"][method]["text"])
            row["asr"][method]["cer"] = character_error_rate(reference, hypothesis)
        if "confidence_gated_method" not in row:
            threshold = float(report.get("confidence_threshold", 0.2))
            row["confidence_gated_method"] = (
                "sbl_adaptive_sector"
                if float(row["srp_phat_confidence"]) >= threshold
                else "delay_sum"
            )

    mean_cer = {
        method: float(np.mean([row["asr"][method]["cer"] for row in rows]))
        for method in method_names
    }
    mean_cer["confidence_gated"] = float(
        np.mean([row["asr"][row["confidence_gated_method"]]["cer"] for row in rows])
    )
    report["mean_cer"] = mean_cer
    report["cer_normalization"] = "tags removed; traditional Chinese converted to simplified"
    report["relative_cer_reduction_vs_input"] = (
        mean_cer["input_channel_0"] - mean_cer["confidence_gated"]
    ) / max(mean_cer["input_channel_0"], np.finfo(float).eps)
    report["adaptive_relative_cer_reduction_vs_input"] = (
        mean_cer["input_channel_0"] - mean_cer["sbl_adaptive_sector"]
    ) / max(mean_cer["input_channel_0"], np.finfo(float).eps)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    compact_report = {key: report[key] for key in report if key != "results"}
    print(json.dumps(compact_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
