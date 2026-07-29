from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy import signal


def _opencv_model_path(path: Path) -> Path:
    """Stage a model under an ASCII path for OpenCV builds with Unicode issues."""

    if os.name != "nt" or str(path).isascii():
        return path
    cache_dir = Path(tempfile.gettempdir()) / "cabin_speech_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / path.name
    if not cached.exists() or cached.stat().st_size != path.stat().st_size:
        shutil.copy2(path, cached)
    return cached


def crop_face_sequence(
    video_path: str | Path,
    start_seconds: float,
    end_seconds: float,
    detector_model_path: str | Path,
    output_size: int = 112,
) -> tuple[np.ndarray, float]:
    """Extract one landmark-informed face crop per frame from an AMI close-up."""

    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    video_path = Path(video_path)
    detector_model_path = Path(detector_model_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video was not found: {video_path}")
    if not detector_model_path.is_file():
        raise FileNotFoundError(f"YuNet model was not found: {detector_model_path}")
    opencv_model_path = _opencv_model_path(detector_model_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
    detector = cv2.FaceDetectorYN.create(
        str(opencv_model_path),
        "",
        (320, 320),
        0.55,
        0.3,
        5000,
    )

    crops: list[np.ndarray] = []
    previous_box: tuple[int, int, int, int] | None = None
    detected_frames = 0
    while capture.get(cv2.CAP_PROP_POS_MSEC) <= end_seconds * 1000.0:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        _, detections = detector.detect(frame)
        if detections is not None and len(detections):
            face = max(detections, key=lambda row: float(row[-1] * row[2] * row[3]))
            x, y, box_width, box_height = (float(value) for value in face[:4])
            detected_frames += 1
            side = 1.25 * max(box_width, box_height)
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            previous_box = (
                round(center_x - side / 2.0),
                round(center_y - side / 2.0),
                round(side),
                round(side),
            )
        if previous_box is None:
            crops.append(np.zeros((output_size, output_size), dtype=np.uint8))
            continue

        x, y, box_width, box_height = previous_box
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + box_width), min(height, y + box_height)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crops.append(cv2.resize(grayscale, (output_size, output_size)))

    capture.release()
    if not crops:
        raise RuntimeError(f"no frames decoded from {video_path}")
    detection_rate = detected_frames / len(crops)
    return np.asarray(crops), float(detection_rate)


class LightASDInference:
    """Inference-only adapter for the official CVPR 2023 Light-ASD weights."""

    def __init__(
        self,
        repository_path: str | Path,
        weights_path: str | Path,
        device: str = "cuda",
    ) -> None:
        import torch

        repository_path = Path(repository_path)
        weights_path = Path(weights_path)
        if not repository_path.is_dir():
            raise FileNotFoundError(f"Light-ASD repository was not found: {repository_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"Light-ASD weights were not found: {weights_path}")
        repository = str(repository_path.resolve())
        if repository not in sys.path:
            sys.path.insert(0, repository)
        from loss import lossAV
        from model.Model import ASD_Model

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = ASD_Model().to(self.device).eval()
        self.classifier = lossAV().to(self.device).eval()

        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        model_state = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
        classifier_state = {
            key.removeprefix("lossAV."): value
            for key, value in state.items()
            if key.startswith("lossAV.")
        }
        self.model.load_state_dict(model_state)
        self.classifier.load_state_dict(classifier_state)

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: int,
        face_frames: np.ndarray,
    ) -> np.ndarray:
        """Return one audible-speaking probability per 25-fps video frame."""

        import python_speech_features
        import torch

        waveform = np.asarray(audio, dtype=np.float64).reshape(-1)
        if sample_rate != 16_000:
            divisor = np.gcd(sample_rate, 16_000)
            waveform = signal.resample_poly(
                waveform,
                16_000 // divisor,
                sample_rate // divisor,
            )
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform_int16 = np.round(waveform * 32767.0).astype(np.int16)
        audio_features = python_speech_features.mfcc(
            waveform_int16,
            16_000,
            numcep=13,
            winlen=0.025,
            winstep=0.010,
        )
        visual_features = np.asarray(face_frames, dtype=np.float32)
        frame_count = min(audio_features.shape[0] // 4, visual_features.shape[0])
        if frame_count < 4:
            raise ValueError("audio-visual clip is too short")
        audio_features = audio_features[: frame_count * 4]
        visual_features = visual_features[:frame_count]

        with torch.no_grad():
            input_audio = torch.from_numpy(audio_features).float().unsqueeze(0).to(self.device)
            input_visual = torch.from_numpy(visual_features).float().unsqueeze(0).to(self.device)
            audio_embedding = self.model.forward_audio_frontend(input_audio)
            visual_embedding = self.model.forward_visual_frontend(input_visual)
            fused = self.model.forward_audio_visual_backend(
                audio_embedding,
                visual_embedding,
            )
            logits = self.classifier.FC(fused.squeeze(1))
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
        return probabilities.detach().cpu().numpy()
