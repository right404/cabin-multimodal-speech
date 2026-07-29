from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMMANDS = {
    "demo": Path("scripts/webcam_mic_demo.py"),
    "evaluate-active-speaker": Path("scripts/evaluate_ami_active_speaker.py"),
    "evaluate-interference": Path("scripts/evaluate_ami_interference.py"),
    "verify-results": Path("scripts/verify_reported_results.py"),
}

HELP = """usage: python -m cabin_speech COMMAND [ARGS...]

commands:
  demo                     webcam + single-microphone real-time demo
  evaluate-active-speaker  AMI Light-ASD active-speaker evaluation
  evaluate-interference    AMI 8-channel directional-interference evaluation
  verify-results           verify saved AMI JSON result consistency
  test                     run the repository pytest suite

Arguments after COMMAND are forwarded to the underlying script.
"""


def build_command(arguments: list[str], project_root: Path = PROJECT_ROOT) -> list[str]:
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        return []
    command, *forwarded = arguments
    if command == "test":
        return [sys.executable, "-m", "pytest", *forwarded]
    if command not in COMMANDS:
        raise ValueError(f"unknown command {command!r}")
    script_path = project_root / COMMANDS[command]
    if not script_path.is_file():
        raise FileNotFoundError(f"command script was not found: {script_path}")
    return [sys.executable, str(script_path), *forwarded]


def main() -> None:
    try:
        command = build_command(sys.argv[1:])
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        print(HELP, file=sys.stderr)
        raise SystemExit(2) from error
    if not command:
        print(HELP)
        return
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    raise SystemExit(completed.returncode)
