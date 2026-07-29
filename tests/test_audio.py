import numpy as np

from cabin_speech.audio import (
    delay_and_sum,
    estimate_linear_array_doa,
    fractional_delay,
    simulate_plane_wave,
)
from cabin_speech.geometry import CabinGeometry


def test_fractional_delay_preserves_shape_and_energy() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate) / sample_rate
    signal = np.sin(2 * np.pi * 700 * time)
    delayed = fractional_delay(signal, 0.00015, sample_rate)
    assert delayed.shape == signal.shape
    assert np.isclose(np.mean(delayed**2), np.mean(signal**2), rtol=0.03)


def test_doa_estimation_on_broadband_signal() -> None:
    sample_rate = 16_000
    rng = np.random.default_rng(7)
    signal = rng.normal(size=sample_rate)
    geometry = CabinGeometry()
    channels = simulate_plane_wave(signal, 30.0, geometry, sample_rate)
    estimate = estimate_linear_array_doa(channels, sample_rate, geometry)
    assert abs(estimate - 30.0) < 5.0


def test_delay_and_sum_returns_one_channel() -> None:
    sample_rate = 16_000
    rng = np.random.default_rng(11)
    signal = rng.normal(size=4000)
    geometry = CabinGeometry()
    channels = simulate_plane_wave(signal, -20.0, geometry, sample_rate)
    enhanced = delay_and_sum(channels, -20.0, geometry, sample_rate)
    assert enhanced.shape == signal.shape
    assert np.isfinite(enhanced).all()
