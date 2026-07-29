import sys
from pathlib import Path

import pytest

from cabin_speech.cli import build_command


def test_build_test_command_forwards_arguments(tmp_path: Path) -> None:
    command = build_command(["test", "-q"], tmp_path)
    assert command == [sys.executable, "-m", "pytest", "-q"]


def test_build_demo_command_uses_repository_script(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "webcam_mic_demo.py"
    script.parent.mkdir()
    script.write_text("", encoding="utf-8")
    command = build_command(["demo", "--camera-index", "1"], tmp_path)
    assert command == [sys.executable, str(script), "--camera-index", "1"]


def test_build_command_rejects_unknown_command(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown command"):
        build_command(["unknown"], tmp_path)
