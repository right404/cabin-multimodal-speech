import numpy as np

from cabin_speech.geometry import CabinGeometry, CircularArrayGeometry


def test_microphone_positions_are_centered() -> None:
    geometry = CabinGeometry(microphone_count=4, microphone_spacing_m=0.05)
    assert np.isclose(geometry.microphone_positions_m.mean(), 0.0)
    assert np.isclose(geometry.array_aperture_m, 0.15)


def test_closest_seat_and_likelihoods() -> None:
    geometry = CabinGeometry()
    assert geometry.closest_seat(-50.0) == 0
    scores = geometry.angle_likelihoods(18.0)
    assert scores.argmax() == 2
    assert np.isclose(scores.sum(), 1.0)


def test_circular_array_positions_and_delays() -> None:
    geometry = CircularArrayGeometry(microphone_count=8, radius_m=0.05)
    positions = geometry.microphone_positions_xy_m
    assert positions.shape == (8, 2)
    assert np.allclose(np.linalg.norm(positions, axis=1), 0.05)
    assert np.allclose(positions.mean(axis=0), 0.0, atol=1e-12)

    delays = geometry.propagation_delays_s(0.0)
    assert delays.shape == (8,)
    assert np.argmax(delays) == 0
