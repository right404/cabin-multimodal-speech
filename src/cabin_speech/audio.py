from __future__ import annotations

import numpy as np

from cabin_speech.geometry import ArrayGeometry, CabinGeometry


def fractional_delay(signal: np.ndarray, delay_s: float, sample_rate: int) -> np.ndarray:
    """Apply a fractional delay using an FFT phase shift.

    The input is padded to reduce circular wrap-around. The returned signal has
    the same length and dtype family as the input.
    """

    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    padding = max(32, int(np.ceil(abs(delay_s) * sample_rate)) + 8)
    padded = np.pad(samples, (padding, padding))
    spectrum = np.fft.rfft(padded)
    frequencies = np.fft.rfftfreq(padded.size, d=1.0 / sample_rate)
    shifted = np.fft.irfft(
        spectrum * np.exp(-2j * np.pi * frequencies * delay_s),
        n=padded.size,
    )
    return shifted[padding : padding + samples.size]


def simulate_plane_wave(
    signal: np.ndarray,
    angle_deg: float,
    geometry: ArrayGeometry,
    sample_rate: int,
) -> np.ndarray:
    """Generate one channel per microphone for a far-field plane wave."""

    delays = geometry.propagation_delays_s(angle_deg)
    channels = [fractional_delay(signal, float(delay), sample_rate) for delay in delays]
    return np.stack(channels, axis=0)


def gcc_phat_delay(
    signal: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    max_delay_s: float | None = None,
    interpolation: int = 16,
) -> float:
    """Estimate the relative delay of ``signal`` against ``reference``."""

    sig = np.asarray(signal, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if sig.ndim != 1 or ref.ndim != 1:
        raise ValueError("signal and reference must be one-dimensional")
    if interpolation < 1:
        raise ValueError("interpolation must be at least one")

    n_fft = sig.size + ref.size
    sig_spec = np.fft.rfft(sig, n=n_fft)
    ref_spec = np.fft.rfft(ref, n=n_fft)
    cross_spectrum = sig_spec * np.conj(ref_spec)
    cross_spectrum /= np.maximum(np.abs(cross_spectrum), np.finfo(float).eps)
    correlation = np.fft.irfft(cross_spectrum, n=interpolation * n_fft)

    max_shift = interpolation * n_fft // 2
    if max_delay_s is not None:
        max_shift = min(max_shift, int(interpolation * sample_rate * max_delay_s))
    correlation = np.concatenate((correlation[-max_shift:], correlation[: max_shift + 1]))
    shift = int(np.argmax(np.abs(correlation))) - max_shift
    return shift / float(interpolation * sample_rate)


def estimate_linear_array_doa(
    multichannel_signal: np.ndarray,
    sample_rate: int,
    geometry: CabinGeometry,
) -> float:
    """Estimate broadside angle using the two end microphones of a ULA."""

    channels = np.asarray(multichannel_signal, dtype=np.float64)
    if channels.ndim != 2:
        raise ValueError("multichannel_signal must have shape [channels, samples]")
    if channels.shape[0] != geometry.microphone_count:
        raise ValueError("channel count does not match geometry")

    maximum_delay = geometry.array_aperture_m / geometry.speed_of_sound_m_s
    delay = gcc_phat_delay(
        channels[-1],
        channels[0],
        sample_rate=sample_rate,
        max_delay_s=maximum_delay,
    )
    sine = np.clip(
        delay * geometry.speed_of_sound_m_s / geometry.array_aperture_m,
        -1.0,
        1.0,
    )
    return float(np.rad2deg(np.arcsin(sine)))


def delay_and_sum(
    multichannel_signal: np.ndarray,
    steering_angle_deg: float,
    geometry: ArrayGeometry,
    sample_rate: int,
) -> np.ndarray:
    """Time-align channels for a target direction and average them."""

    channels = np.asarray(multichannel_signal, dtype=np.float64)
    if channels.shape[0] != geometry.microphone_count:
        raise ValueError("channel count does not match geometry")
    delays = geometry.propagation_delays_s(steering_angle_deg)
    aligned = [
        fractional_delay(channel, -float(delay), sample_rate)
        for channel, delay in zip(channels, delays, strict=True)
    ]
    return np.mean(np.stack(aligned, axis=0), axis=0)
