from __future__ import annotations

import math
import queue
import threading
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import sounddevice as sd


def rms_dbfs(samples: np.ndarray) -> float:
    """Return RMS level in dBFS for floating-point audio in [-1, 1]."""

    values = np.asarray(samples, dtype=np.float32)
    if values.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    return max(20.0 * math.log10(max(rms, 1e-6)), -120.0)


@dataclass
class AudioState:
    dbfs: float = -120.0
    noise_floor_dbfs: float = -60.0
    threshold_dbfs: float = -51.0
    speech_probability: float = 0.0
    is_speech: bool = False
    calibrated: bool = False
    overflow_count: int = 0


@dataclass(frozen=True)
class SpeechSegment:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / self.sample_rate


class SpeechSegmenter:
    """Turn block-level VAD decisions into utterances with a short pre-roll."""

    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        pre_roll_ms: float = 250.0,
        minimum_seconds: float = 0.6,
        maximum_seconds: float = 15.0,
    ) -> None:
        if sample_rate <= 0 or block_size <= 0:
            raise ValueError("sample_rate and block_size must be positive")
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        pre_roll_blocks = max(1, round((pre_roll_ms / 1000.0) * self.sample_rate / self.block_size))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_blocks)
        self._active_blocks: list[np.ndarray] = []
        self._minimum_samples = round(minimum_seconds * self.sample_rate)
        self._maximum_samples = round(maximum_seconds * self.sample_rate)

    def reset(self) -> None:
        self._pre_roll.clear()
        self._active_blocks.clear()

    def _finish(self) -> SpeechSegment | None:
        if not self._active_blocks:
            return None
        samples = np.concatenate(self._active_blocks).astype(np.float32, copy=False)
        self._active_blocks.clear()
        if samples.size < self._minimum_samples:
            return None
        return SpeechSegment(samples=samples, sample_rate=self.sample_rate)

    def process(self, samples: np.ndarray, state: AudioState) -> SpeechSegment | None:
        block = np.asarray(samples, dtype=np.float32).copy()
        if block.ndim != 1:
            raise ValueError("audio block must be one-dimensional")
        if not state.calibrated:
            self._pre_roll.append(block)
            return None

        if state.is_speech:
            if not self._active_blocks:
                self._active_blocks.extend(self._pre_roll)
                self._pre_roll.clear()
            self._active_blocks.append(block)
            active_samples = sum(item.size for item in self._active_blocks)
            if active_samples >= self._maximum_samples:
                return self._finish()
            return None

        segment = self._finish()
        self._pre_roll.append(block)
        return segment


class AdaptiveEnergyVAD:
    """Low-latency energy VAD with startup calibration and hangover."""

    def __init__(
        self,
        calibration_frames: int = 40,
        threshold_margin_db: float = 9.0,
        hangover_frames: int = 7,
    ) -> None:
        if calibration_frames < 1:
            raise ValueError("calibration_frames must be positive")
        self.calibration_frames = int(calibration_frames)
        self.threshold_margin_db = float(threshold_margin_db)
        self.hangover_frames = int(hangover_frames)
        self._calibration_levels: deque[float] = deque(maxlen=self.calibration_frames)
        self._noise_floor = -60.0
        self._hangover = 0

    def reset(self) -> None:
        self._calibration_levels.clear()
        self._noise_floor = -60.0
        self._hangover = 0

    def process(self, samples: np.ndarray) -> AudioState:
        level = rms_dbfs(samples)
        if len(self._calibration_levels) < self.calibration_frames:
            self._calibration_levels.append(level)
            robust_level = float(np.percentile(self._calibration_levels, 65))
            self._noise_floor = float(np.clip(robust_level, -80.0, -25.0))
            threshold = self._noise_floor + self.threshold_margin_db
            return AudioState(
                dbfs=level,
                noise_floor_dbfs=self._noise_floor,
                threshold_dbfs=threshold,
                calibrated=False,
            )

        threshold = self._noise_floor + self.threshold_margin_db
        logit = float(np.clip((level - threshold) / 3.0, -20.0, 20.0))
        probability = 1.0 / (1.0 + math.exp(-logit))
        raw_speech = probability >= 0.55

        if raw_speech:
            self._hangover = self.hangover_frames
        elif self._hangover > 0:
            self._hangover -= 1
            probability = max(probability, 0.58)

        is_speech = raw_speech or self._hangover > 0
        if not is_speech and level < threshold:
            self._noise_floor = float(
                np.clip(0.985 * self._noise_floor + 0.015 * level, -80.0, -25.0)
            )
        return AudioState(
            dbfs=level,
            noise_floor_dbfs=self._noise_floor,
            threshold_dbfs=self._noise_floor + self.threshold_margin_db,
            speech_probability=float(np.clip(probability, 0.0, 1.0)),
            is_speech=is_speech,
            calibrated=True,
        )


class AudioLevelMonitor:
    """Thread-safe microphone monitor backed by a sounddevice callback."""

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 16_000,
        block_size: int = 800,
        calibration_seconds: float = 2.0,
        threshold_margin_db: float = 9.0,
        hangover_ms: float = 350.0,
        segment_pre_roll_ms: float = 250.0,
        segment_min_seconds: float = 0.6,
        segment_max_seconds: float = 15.0,
    ) -> None:
        block_duration = block_size / sample_rate
        calibration_frames = max(1, round(calibration_seconds / block_duration))
        hangover_frames = max(0, round((hangover_ms / 1000.0) / block_duration))
        self.device = device
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.vad = AdaptiveEnergyVAD(
            calibration_frames=calibration_frames,
            threshold_margin_db=threshold_margin_db,
            hangover_frames=hangover_frames,
        )
        self._state = AudioState()
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._segmenter = SpeechSegmenter(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            pre_roll_ms=segment_pre_roll_ms,
            minimum_seconds=segment_min_seconds,
            maximum_seconds=segment_max_seconds,
        )
        self._segments: queue.Queue[SpeechSegment] = queue.Queue(maxsize=4)

    def _callback(
        self,
        input_data: np.ndarray,
        _frames: int,
        _time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        mono = np.mean(input_data, axis=1)
        state = self.vad.process(mono)
        if status.input_overflow:
            state.overflow_count = self._state.overflow_count + 1
        else:
            state.overflow_count = self._state.overflow_count
        segment = self._segmenter.process(mono, state)
        if segment is not None:
            if self._segments.full():
                try:
                    self._segments.get_nowait()
                except queue.Empty:
                    pass
            self._segments.put_nowait(segment)
        with self._lock:
            self._state = state

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def recalibrate(self) -> None:
        self.vad.reset()
        self._segmenter.reset()
        with self._lock:
            self._state = AudioState()

    def pop_segment(self) -> SpeechSegment | None:
        try:
            return self._segments.get_nowait()
        except queue.Empty:
            return None

    def state(self) -> AudioState:
        with self._lock:
            return replace(self._state)

    def __enter__(self) -> AudioLevelMonitor:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


def live_fusion_probabilities(
    speech_probability: float,
    face_motion_scores: dict[int, float],
) -> dict[str, float]:
    """Fuse global microphone VAD with per-face mouth motion.

    With a single microphone, audio can indicate when speech is present but
    cannot localise the source. Visual motion distributes speech probability
    among visible faces and leaves residual mass for an off-camera speaker.
    """

    speech = float(np.clip(speech_probability, 0.0, 1.0))
    result: dict[str, float] = {"no_speech": 1.0 - speech}
    if not face_motion_scores:
        result["off_camera"] = speech
        return result

    track_ids = list(face_motion_scores)
    motions = np.asarray([face_motion_scores[track_id] for track_id in track_ids])
    motions = np.clip(motions, 0.0, 1.0)
    visual_evidence = float(np.clip(motions.max() * 2.5, 0.0, 1.0))
    visible_mass = speech * visual_evidence
    result["off_camera"] = speech - visible_mass

    logits = 6.0 * motions
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    for track_id, probability in zip(track_ids, visible_mass * weights, strict=True):
        result[f"face_{track_id}"] = float(probability)
    return result
