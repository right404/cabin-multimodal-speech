from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cabin_speech.asr import AsyncASRWorker, TranscriptResult, WhisperASR
from cabin_speech.config import load_config
from cabin_speech.realtime import AudioLevelMonitor, live_fusion_probabilities
from cabin_speech.vision import FaceMouthMotionTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam + microphone active-speaker demo.")
    parser.add_argument("--config", default="configs/realtime.yaml")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--audio-device", type=int)
    parser.add_argument(
        "--headless-seconds",
        type=float,
        default=0.0,
        help="Run hardware checks without opening a window.",
    )
    parser.add_argument("--enable-asr", action="store_true")
    parser.add_argument("--asr-model")
    parser.add_argument(
        "--font-path",
        type=Path,
        help="Optional TrueType font for Chinese subtitles.",
    )
    parser.add_argument("--save-transcripts", action="store_true")
    return parser.parse_args()


def overlay_status(
    frame,
    audio_state,
    observations,
    probabilities: dict[str, float],
    asr_busy: bool = False,
) -> None:
    winner = max(probabilities, key=probabilities.get)
    speech_text = "SPEECH" if audio_state.is_speech else "SILENCE"
    status_color = (40, 210, 40) if audio_state.is_speech else (160, 160, 160)
    cv2.putText(
        frame,
        (
            f"{speech_text}  level={audio_state.dbfs:.1f} dBFS  "
            f"noise={audio_state.noise_floor_dbfs:.1f} dBFS"
        ),
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        status_color,
        2,
        cv2.LINE_AA,
    )
    if not audio_state.calibrated:
        cv2.putText(
            frame,
            "Calibrating microphone noise floor - please stay quiet",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    for observation in observations:
        key = f"face_{observation.track_id}"
        probability = probabilities.get(key, 0.0)
        active = winner == key and probability > 0.35
        color = (30, 220, 30) if active else (220, 160, 30)
        x, y, width, height = observation.bbox
        mx, my, mw, mh = observation.mouth_bbox
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (180, 100, 255), 1)
        cv2.putText(
            frame,
            (
                f"face {observation.track_id}  active={probability:.2f}  "
                f"lip={observation.mouth_motion:.2f}"
            ),
            (x, max(20, y - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

    off_camera = probabilities.get("off_camera", 0.0)
    cv2.putText(
        frame,
        (
            f"off-camera={off_camera:.2f}  ASR={'RUNNING' if asr_busy else 'READY'}  "
            "[Q] quit  [C] recalibrate  [S] screenshot"
        ),
        (18, frame.shape[0] - 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def draw_chinese_subtitle(
    frame: np.ndarray,
    transcript: TranscriptResult | None,
    font_path: Path | None,
) -> np.ndarray:
    if transcript is None or not transcript.text:
        return frame
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 64), (width, height), (10, 10, 10), -1)
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = (
        ImageFont.truetype(str(font_path), 24)
        if font_path is not None
        else ImageFont.load_default()
    )
    metrics_font = (
        ImageFont.truetype(str(font_path), 15)
        if font_path is not None
        else ImageFont.load_default()
    )
    draw.text((18, height - 58), transcript.text[:42], font=font, fill=(255, 255, 255))
    metrics = (
        f"{transcript.audio_duration_seconds:.1f}s / "
        f"{transcript.latency_seconds:.2f}s / RTF {transcript.real_time_factor:.2f}"
    )
    draw.text((width - 250, height - 23), metrics, font=metrics_font, fill=(170, 220, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def resolve_subtitle_font(explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"subtitle font was not found: {explicit_path}")
        return explicit_path
    windows_root = os.environ.get("WINDIR")
    if windows_root:
        candidate = Path(windows_root) / "Fonts" / "msyh.ttc"
        if candidate.is_file():
            return candidate
    return None


def append_transcript_log(path: Path, transcript: TranscriptResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "text": transcript.text,
        "audio_duration_seconds": transcript.audio_duration_seconds,
        "latency_seconds": transcript.latency_seconds,
        "real_time_factor": transcript.real_time_factor,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    camera_config = config["camera"]
    audio_config = config["audio"]
    asr_config = config["asr"]
    subtitle_font = resolve_subtitle_font(args.font_path)
    camera_index = (
        int(args.camera_index) if args.camera_index is not None else int(camera_config["index"])
    )
    audio_device = args.audio_device if args.audio_device is not None else audio_config["device"]

    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config["width"]))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config["height"]))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera index {camera_index}")

    tracker = FaceMouthMotionTracker(
        scale_factor=float(camera_config["detection_scale_factor"]),
        min_neighbors=int(camera_config["detection_min_neighbors"]),
        minimum_face_size_px=int(camera_config["minimum_face_size_px"]),
    )
    monitor = AudioLevelMonitor(
        device=audio_device,
        sample_rate=int(audio_config["sample_rate"]),
        block_size=int(audio_config["block_size"]),
        calibration_seconds=float(audio_config["calibration_seconds"]),
        threshold_margin_db=float(audio_config["threshold_margin_db"]),
        hangover_ms=float(audio_config["hangover_ms"]),
        segment_pre_roll_ms=float(audio_config["segment_pre_roll_ms"]),
        segment_min_seconds=float(audio_config["segment_min_seconds"]),
        segment_max_seconds=float(audio_config["segment_max_seconds"]),
    )

    asr_worker = None
    transcript_history: deque[TranscriptResult] = deque(maxlen=3)
    transcript_log_path = (
        PROJECT_ROOT
        / "artifacts"
        / "transcripts"
        / f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    if args.enable_asr:
        model_id = args.asr_model or str(asr_config["model"])
        print(f"loading_asr_model={model_id}")
        backend = WhisperASR(
            model_id=model_id,
            language=str(asr_config["language"]),
            device=str(asr_config["device"]),
            max_new_tokens=int(asr_config["max_new_tokens"]),
        )
        asr_worker = AsyncASRWorker(backend)
        print("asr_model=ready")

    started = time.monotonic()
    frame_count = 0
    detected_face_frames = 0
    try:
        monitor.start()
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera opened but returned no frame")
            frame_count += 1
            observations = tracker.update(frame)
            detected_face_frames += int(bool(observations))
            audio_state = monitor.state()
            probabilities = live_fusion_probabilities(
                audio_state.speech_probability,
                {observation.track_id: observation.mouth_motion for observation in observations},
            )
            if asr_worker is not None:
                while (segment := monitor.pop_segment()) is not None:
                    asr_worker.submit(segment)
                if (transcript := asr_worker.poll()) is not None:
                    transcript_history.append(transcript)
                    print(
                        f"transcript={transcript.text!r} "
                        f"audio_s={transcript.audio_duration_seconds:.2f} "
                        f"latency_s={transcript.latency_seconds:.2f} "
                        f"rtf={transcript.real_time_factor:.3f}"
                    )
                    if args.save_transcripts:
                        append_transcript_log(transcript_log_path, transcript)

            overlay_status(
                frame,
                audio_state,
                observations,
                probabilities,
                asr_busy=asr_worker.busy if asr_worker is not None else False,
            )
            frame = draw_chinese_subtitle(
                frame,
                transcript_history[-1] if transcript_history else None,
                subtitle_font,
            )

            elapsed = time.monotonic() - started
            if args.headless_seconds > 0:
                if elapsed >= args.headless_seconds:
                    print(
                        f"hardware_check=ok camera_index={camera_index} "
                        f"frames={frame_count} face_frames={detected_face_frames} "
                        f"audio_calibrated={audio_state.calibrated} "
                        f"audio_level_dbfs={audio_state.dbfs:.1f}"
                    )
                    break
                continue

            cv2.imshow("Cabin audio-visual active-speaker demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                monitor.recalibrate()
            if key == ord("s"):
                screenshot_dir = PROJECT_ROOT / "artifacts" / "screenshots"
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                output = screenshot_dir / f"demo-{int(time.time())}.jpg"
                cv2.imwrite(str(output), frame)
                print(f"saved={output}")
    finally:
        monitor.stop()
        if asr_worker is not None:
            asr_worker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
