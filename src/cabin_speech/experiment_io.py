from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found: {path}\n"
            "See README.md -> Data preparation for the required download layout."
        )
    return path


def require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{description} was not found: {path}\n"
            "See README.md -> Data preparation for the required download layout."
        )
    return path


def read_active_speaker_prior(
    result_path: str | Path,
    target_start_seconds: float,
    tolerance_seconds: float = 1e-3,
) -> dict[str, Any]:
    """Read the persisted Light-ASD decision for one evaluated AMI segment."""

    path = Path(result_path)
    require_file(path, "active-speaker result JSON")
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"{path} does not contain a segments list")
    matches = [
        segment
        for segment in segments
        if isinstance(segment, dict)
        and abs(float(segment["start"]) - target_start_seconds) <= tolerance_seconds
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one active-speaker result at {target_start_seconds:.3f}s, "
            f"found {len(matches)} in {path}"
        )
    result = matches[0]
    if not isinstance(result.get("prediction"), str):
        raise ValueError(f"active-speaker result at {target_start_seconds:.3f}s has no prediction")
    return result
