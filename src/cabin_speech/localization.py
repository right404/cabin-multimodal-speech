from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from cabin_speech.geometry import ArrayGeometry
from cabin_speech.sbl_mvdr import steering_dictionary


@dataclass(frozen=True)
class DOAResult:
    angle_deg: float
    confidence: float
    angles_deg: np.ndarray
    scores: np.ndarray


def estimate_srp_phat_doa(
    multichannel_audio: np.ndarray,
    sample_rate: int,
    geometry: ArrayGeometry,
    angles_deg: np.ndarray | None = None,
    n_fft: int = 512,
    hop_length: int = 128,
    min_frequency_hz: float = 300.0,
    max_frequency_hz: float = 4_000.0,
) -> DOAResult:
    """Estimate broadband DOA with a steered-response PHAT spatial scan."""

    audio = np.asarray(multichannel_audio, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("multichannel_audio must have shape [channels, samples]")
    if audio.shape[0] != geometry.microphone_count:
        raise ValueError("channel count does not match geometry")
    if audio.shape[1] < n_fft:
        raise ValueError("audio must contain at least n_fft samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if angles_deg is None:
        angles = np.arange(-180.0, 180.0, 2.0)
    else:
        angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("angles_deg must be a non-empty one-dimensional array")

    frequencies, _, spectra = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        axis=-1,
        boundary=None,
        padded=False,
    )
    eligible = np.flatnonzero(
        (frequencies >= min_frequency_hz)
        & (frequencies <= min(max_frequency_hz, sample_rate / 2.0))
    )
    if eligible.size == 0:
        raise ValueError("no STFT bins fall inside the configured frequency range")

    scores = np.zeros(angles.size, dtype=np.float64)
    epsilon = np.finfo(float).eps
    for frequency_index in eligible:
        phase_only = spectra[:, frequency_index, :] / np.maximum(
            np.abs(spectra[:, frequency_index, :]),
            epsilon,
        )
        dictionary = steering_dictionary(
            geometry,
            angles,
            frequencies[frequency_index],
        )
        response = dictionary.conj().T @ phase_only / geometry.microphone_count
        scores += np.mean(np.abs(response) ** 2, axis=1)
    scores /= eligible.size

    peak_index = int(np.argmax(scores))
    median_score = float(np.median(scores))
    peak_score = float(scores[peak_index])
    confidence = (peak_score - median_score) / max(peak_score, epsilon)
    return DOAResult(
        angle_deg=float(angles[peak_index]),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        angles_deg=angles,
        scores=scores,
    )
