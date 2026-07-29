from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class MultichannelRecording:
    """A time-aligned multichannel waveform in channels-first layout."""

    sample_rate: int
    audio: np.ndarray
    source_path: Path

    @property
    def duration_seconds(self) -> float:
        return self.audio.shape[1] / self.sample_rate


def load_multichannel_audio(
    path: str | Path,
    expected_channels: int | None = None,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> MultichannelRecording:
    """Read synchronized WAV/FLAC audio and return ``[channels, samples]``.

    The decoder seeks before reading so a short segment can be loaded without
    decoding or copying the full meeting recording.
    """

    source_path = Path(path)
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    with sf.SoundFile(source_path) as audio_file:
        sample_rate = int(audio_file.samplerate)
        if audio_file.channels < 2:
            raise ValueError(f"{source_path.name} is not a multichannel recording")
        if expected_channels is not None and audio_file.channels != expected_channels:
            raise ValueError(f"expected {expected_channels} channels, found {audio_file.channels}")
        start_sample = int(round(start_seconds * sample_rate))
        if start_sample >= audio_file.frames:
            raise ValueError("start_seconds lies beyond the end of the recording")
        audio_file.seek(start_sample)
        if duration_seconds is None:
            frame_count = -1
        else:
            frame_count = int(round(duration_seconds * sample_rate))
        segment = audio_file.read(
            frames=frame_count,
            dtype="float64",
            always_2d=True,
        )
    return MultichannelRecording(
        sample_rate=sample_rate,
        audio=np.ascontiguousarray(segment.T),
        source_path=source_path,
    )


def load_multichannel_wav(
    path: str | Path,
    expected_channels: int | None = None,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> MultichannelRecording:
    """Backward-compatible alias for :func:`load_multichannel_audio`."""

    return load_multichannel_audio(
        path=path,
        expected_channels=expected_channels,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )


def load_synchronised_mono_files(
    paths: list[str | Path],
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> MultichannelRecording:
    """Stack time-aligned mono files into a channels-first recording."""

    if not paths:
        raise ValueError("at least one mono audio path is required")
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    source_paths = [Path(path) for path in paths]
    channel_segments: list[np.ndarray] = []
    expected_sample_rate: int | None = None
    expected_total_frames: int | None = None
    expected_segment_frames: int | None = None

    for source_path in source_paths:
        with sf.SoundFile(source_path) as audio_file:
            if audio_file.channels != 1:
                raise ValueError(f"{source_path.name} is not mono")
            sample_rate = int(audio_file.samplerate)
            total_frames = int(audio_file.frames)
            if expected_sample_rate is None:
                expected_sample_rate = sample_rate
                expected_total_frames = total_frames
            elif sample_rate != expected_sample_rate:
                raise ValueError("mono files have different sample rates")
            elif total_frames != expected_total_frames:
                raise ValueError("mono files have different frame counts")

            start_sample = int(round(start_seconds * sample_rate))
            if start_sample >= total_frames:
                raise ValueError("start_seconds lies beyond the end of the recording")
            audio_file.seek(start_sample)
            frame_count = (
                -1 if duration_seconds is None else int(round(duration_seconds * sample_rate))
            )
            segment = audio_file.read(frames=frame_count, dtype="float64")
            if expected_segment_frames is None:
                expected_segment_frames = int(segment.size)
            elif segment.size != expected_segment_frames:
                raise ValueError("mono files yielded different segment lengths")
            channel_segments.append(segment)

    assert expected_sample_rate is not None
    return MultichannelRecording(
        sample_rate=expected_sample_rate,
        audio=np.ascontiguousarray(np.stack(channel_segments)),
        source_path=source_paths[0],
    )


def discover_aishell4_recordings(root: str | Path) -> list[Path]:
    """Find AISHELL-4 meeting WAV/FLAC files below an extracted corpus root."""

    root_path = Path(root)
    supported_suffixes = {".wav", ".flac"}
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )
