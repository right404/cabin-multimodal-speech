import numpy as np

from cabin_speech.audio import simulate_plane_wave
from cabin_speech.geometry import CircularArrayGeometry
from cabin_speech.localization import estimate_srp_phat_doa


def test_srp_phat_localizes_circular_array_plane_wave() -> None:
    sample_rate = 16_000
    rng = np.random.default_rng(71)
    source = rng.normal(size=8_000)
    geometry = CircularArrayGeometry(microphone_count=8, radius_m=0.05)
    channels = simulate_plane_wave(source, 64.0, geometry, sample_rate)
    channels += rng.normal(scale=0.02, size=channels.shape)

    result = estimate_srp_phat_doa(
        channels,
        sample_rate,
        geometry,
        angles_deg=np.arange(-180.0, 180.0, 2.0),
    )
    assert abs(result.angle_deg - 64.0) <= 2.0
    assert result.confidence > 0.2
