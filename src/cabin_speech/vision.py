from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceObservation:
    track_id: int
    bbox: tuple[int, int, int, int]
    mouth_bbox: tuple[int, int, int, int]
    mouth_motion: float
    visibility: float


@dataclass
class _FaceTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    previous_mouth: np.ndarray | None = None
    motion_ema: float = 0.0
    missed_frames: int = 0


def intersection_over_union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def normalised_mouth_motion(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 0.0
    if previous.shape != current.shape:
        previous = cv2.resize(previous, (current.shape[1], current.shape[0]))
    difference = cv2.absdiff(previous, current)
    raw_motion = float(np.mean(difference) / 255.0)
    return float(np.clip((raw_motion - 0.008) / 0.075, 0.0, 1.0))


class FaceMouthMotionTracker:
    """Haar face detector plus lower-face frame-difference lip-motion baseline."""

    def __init__(
        self,
        scale_factor: float = 1.12,
        min_neighbors: int = 5,
        minimum_face_size_px: int = 80,
        maximum_missed_frames: int = 8,
    ) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"unable to load face detector: {cascade_path}")
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.minimum_face_size_px = int(minimum_face_size_px)
        self.maximum_missed_frames = int(maximum_missed_frames)
        self._tracks: dict[int, _FaceTrack] = {}
        self._next_track_id = 1

    @staticmethod
    def _mouth_region(
        grayscale: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        x, y, width, height = bbox
        mouth_x = x + round(0.16 * width)
        mouth_y = y + round(0.58 * height)
        mouth_width = max(1, round(0.68 * width))
        mouth_height = max(1, round(0.30 * height))
        roi = grayscale[
            mouth_y : mouth_y + mouth_height,
            mouth_x : mouth_x + mouth_width,
        ]
        if roi.size == 0:
            roi = np.zeros((32, 64), dtype=np.uint8)
        roi = cv2.resize(roi, (64, 32))
        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        return roi, (mouth_x, mouth_y, mouth_width, mouth_height)

    def _match_track(self, bbox: tuple[int, int, int, int], used: set[int]) -> _FaceTrack:
        candidates = [
            (intersection_over_union(track.bbox, bbox), track)
            for track in self._tracks.values()
            if track.track_id not in used
        ]
        if candidates:
            best_iou, best_track = max(candidates, key=lambda item: item[0])
            if best_iou >= 0.20:
                return best_track
        track = _FaceTrack(track_id=self._next_track_id, bbox=bbox)
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def update(self, frame_bgr: np.ndarray) -> list[FaceObservation]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be a BGR image")
        grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)
        detected = self.detector.detectMultiScale(
            grayscale,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.minimum_face_size_px, self.minimum_face_size_px),
        )
        bboxes = [tuple(map(int, bbox)) for bbox in detected]
        bboxes.sort(key=lambda bbox: bbox[0])

        for track in self._tracks.values():
            track.missed_frames += 1
        used: set[int] = set()
        observations = []
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        for bbox in bboxes:
            track = self._match_track(bbox, used)
            used.add(track.track_id)
            mouth, mouth_bbox = self._mouth_region(grayscale, bbox)
            instantaneous = normalised_mouth_motion(track.previous_mouth, mouth)
            track.motion_ema = 0.68 * track.motion_ema + 0.32 * instantaneous
            track.previous_mouth = mouth
            track.bbox = bbox
            track.missed_frames = 0
            visibility = float(np.clip((bbox[2] * bbox[3]) / (0.12 * frame_area), 0.0, 1.0))
            observations.append(
                FaceObservation(
                    track_id=track.track_id,
                    bbox=bbox,
                    mouth_bbox=mouth_bbox,
                    mouth_motion=track.motion_ema,
                    visibility=visibility,
                )
            )

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_frames > self.maximum_missed_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return observations


class YuNetMouthMotionTracker:
    """Landmark-aligned mouth motion for a single-person close-up video."""

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.65,
        smoothing: float = 0.68,
    ) -> None:
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold,
            0.3,
            5000,
        )
        self.smoothing = float(smoothing)
        self.previous_mouth: np.ndarray | None = None
        self.motion_ema = 0.0

    @staticmethod
    def _aligned_mouth(
        grayscale: np.ndarray,
        face: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        source = np.asarray(
            [
                [face[4], face[5]],
                [face[6], face[7]],
                [face[8], face[9]],
            ],
            dtype=np.float32,
        )
        destination = np.asarray([[21, 22], [43, 22], [32, 38]], dtype=np.float32)
        transform = cv2.getAffineTransform(source, destination)
        aligned = cv2.warpAffine(
            grayscale,
            transform,
            (64, 64),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mouth = cv2.GaussianBlur(aligned[39:63, 10:54], (3, 3), 0)

        mouth_points = np.asarray(
            [[face[10], face[11]], [face[12], face[13]]],
            dtype=np.float64,
        )
        center = mouth_points.mean(axis=0)
        face_width = float(face[2])
        face_height = float(face[3])
        mouth_width = max(1, round(0.55 * face_width))
        mouth_height = max(1, round(0.24 * face_height))
        mouth_bbox = (
            round(center[0] - mouth_width / 2),
            round(center[1] - mouth_height / 2),
            mouth_width,
            mouth_height,
        )
        return mouth, mouth_bbox

    def update(self, frame_bgr: np.ndarray) -> FaceObservation | None:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame must be a BGR image")
        height, width = frame_bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(frame_bgr)
        if detections is None or len(detections) == 0:
            return None

        face = max(detections, key=lambda row: float(row[-1] * row[2] * row[3]))
        grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mouth, mouth_bbox = self._aligned_mouth(grayscale, face)
        instantaneous = normalised_mouth_motion(self.previous_mouth, mouth)
        self.motion_ema = self.smoothing * self.motion_ema + (1.0 - self.smoothing) * instantaneous
        self.previous_mouth = mouth

        bbox = tuple(round(float(value)) for value in face[:4])
        frame_area = height * width
        visibility = float(np.clip((bbox[2] * bbox[3]) / (0.12 * frame_area), 0.0, 1.0))
        return FaceObservation(
            track_id=1,
            bbox=bbox,
            mouth_bbox=mouth_bbox,
            mouth_motion=self.motion_ema,
            visibility=visibility,
        )
