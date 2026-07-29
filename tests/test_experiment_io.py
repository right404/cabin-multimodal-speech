import json
from pathlib import Path

import pytest

from cabin_speech.experiment_io import read_active_speaker_prior


def test_read_active_speaker_prior_selects_requested_segment(tmp_path: Path) -> None:
    result_path = tmp_path / "active.json"
    result_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 1.0, "prediction": "A"},
                    {"start": 2.0, "prediction": "B"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = read_active_speaker_prior(result_path, 2.0)
    assert result["prediction"] == "B"


def test_read_active_speaker_prior_rejects_missing_segment(tmp_path: Path) -> None:
    result_path = tmp_path / "active.json"
    result_path.write_text('{"segments": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="found 0"):
        read_active_speaker_prior(result_path, 2.0)
