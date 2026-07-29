from __future__ import annotations

import argparse
import json

import cv2
import sounddevice as sd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List available cameras and audio inputs.")
    parser.add_argument("--max-camera-index", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = []
    for index in range(args.max_camera_index + 1):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        opened = capture.isOpened()
        frame_ok = False
        width = 0
        height = 0
        if opened:
            frame_ok, frame = capture.read()
            if frame_ok:
                height, width = frame.shape[:2]
        capture.release()
        if opened:
            cameras.append(
                {
                    "index": index,
                    "frame_ok": bool(frame_ok),
                    "width": width,
                    "height": height,
                }
            )

    audio_inputs = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            audio_inputs.append(
                {
                    "index": index,
                    "name": str(device["name"]),
                    "max_input_channels": int(device["max_input_channels"]),
                    "default_samplerate": float(device["default_samplerate"]),
                }
            )
    report = {
        "default_audio_input": int(sd.default.device[0]),
        "cameras": cameras,
        "audio_inputs": audio_inputs,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
