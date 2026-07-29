from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from cabin_speech.datasets import load_multichannel_wav, load_synchronised_mono_files


def test_load_multichannel_wav_returns_channels_first(tmp_path: Path) -> None:
    sample_rate = 16_000
    samples = np.arange(1_600 * 8, dtype=np.int16).reshape(1_600, 8)
    path = tmp_path / "meeting.wav"
    wavfile.write(path, sample_rate, samples)

    recording = load_multichannel_wav(
        path,
        expected_channels=8,
        start_seconds=0.02,
        duration_seconds=0.05,
    )
    assert recording.sample_rate == sample_rate
    assert recording.audio.shape == (8, 800)
    assert recording.duration_seconds == 0.05
    assert np.max(np.abs(recording.audio)) <= 1.0


def test_load_multichannel_wav_rejects_wrong_channel_count(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    wavfile.write(path, 8_000, np.zeros((100, 2), dtype=np.int16))
    with pytest.raises(ValueError, match="expected 8 channels"):
        load_multichannel_wav(path, expected_channels=8)


def test_load_synchronised_mono_files_stacks_channels(tmp_path: Path) -> None:
    first = tmp_path / "ch1.wav"
    second = tmp_path / "ch2.wav"
    wavfile.write(first, 8_000, np.full(800, 1000, dtype=np.int16))
    wavfile.write(second, 8_000, np.full(800, -1000, dtype=np.int16))

    recording = load_synchronised_mono_files(
        [first, second],
        start_seconds=0.02,
        duration_seconds=0.05,
    )
    assert recording.sample_rate == 8_000
    assert recording.audio.shape == (2, 400)
    assert np.all(recording.audio[0] > 0)
    assert np.all(recording.audio[1] < 0)
