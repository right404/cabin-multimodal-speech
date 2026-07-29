import numpy as np

from cabin_speech.geometry import CabinGeometry, CircularArrayGeometry
from cabin_speech.metrics import (
    character_error_rate,
    normalize_chinese_text,
    scale_invariant_sdr,
)
from cabin_speech.sbl_mvdr import (
    INCMConfig,
    SBLConfig,
    WidebandConfig,
    adaptive_sector_half_width,
    reconstruct_incm_mvdr,
    run_mmv_sbl,
    steering_dictionary,
    wideband_sbl_mvdr,
)


def test_scale_invariant_sdr_rewards_a_cleaner_estimate() -> None:
    rng = np.random.default_rng(5)
    reference = rng.normal(size=1_000)
    noisy = reference + rng.normal(scale=0.8, size=reference.shape)
    cleaner = reference + rng.normal(scale=0.1, size=reference.shape)
    assert scale_invariant_sdr(reference, cleaner) > scale_invariant_sdr(reference, noisy)


def test_character_error_rate() -> None:
    assert character_error_rate("今天天气", "今天晴天") == 0.5
    assert character_error_rate("测试", "测试") == 0.0
    assert normalize_chinese_text("這次，<sil>測試 A-1") == "这次测试a1"


def test_adaptive_sector_is_narrower_for_confident_visual_prior() -> None:
    config = WidebandConfig()
    assert adaptive_sector_half_width(None, config) == 5.0
    assert adaptive_sector_half_width(0.9, config) < adaptive_sector_half_width(0.2, config)


def test_circular_array_dictionary_has_unit_magnitude() -> None:
    geometry = CircularArrayGeometry(microphone_count=8, radius_m=0.05)
    dictionary = steering_dictionary(
        geometry,
        np.arange(-180.0, 180.0, 15.0),
        frequency_hz=2_000.0,
    )
    assert dictionary.shape == (8, 24)
    assert np.allclose(np.abs(dictionary), 1.0)


def test_sbl_incm_finds_target_and_suppresses_interferer() -> None:
    rng = np.random.default_rng(19)
    # The paper's reference array uses M=10 and half-wavelength spacing.
    geometry = CabinGeometry(microphone_count=10, microphone_spacing_m=0.08)
    angles = np.arange(-60.0, 61.0, 2.0)
    dictionary = steering_dictionary(geometry, angles, frequency_hz=2_000.0)
    target = steering_dictionary(geometry, np.asarray([10.0]), 2_000.0)[:, 0]
    interferer = steering_dictionary(geometry, np.asarray([-34.0]), 2_000.0)[:, 0]
    snapshots = 80
    desired_signal = (rng.normal(size=snapshots) + 1j * rng.normal(size=snapshots)) / np.sqrt(2.0)
    interference_signal = (
        3.0 * (rng.normal(size=snapshots) + 1j * rng.normal(size=snapshots)) / np.sqrt(2.0)
    )
    noise = (
        0.08
        * (
            rng.normal(size=(geometry.microphone_count, snapshots))
            + 1j * rng.normal(size=(geometry.microphone_count, snapshots))
        )
        / np.sqrt(2.0)
    )
    mixture = (
        target[:, np.newaxis] * desired_signal
        + interferer[:, np.newaxis] * interference_signal
        + noise
    )

    sbl = run_mmv_sbl(
        mixture,
        dictionary,
        SBLConfig(max_iterations=100, tolerance=1e-4),
    )
    result = reconstruct_incm_mvdr(
        sbl.power_spectrum,
        sbl.noise_variance,
        dictionary,
        angles,
        nominal_target_angle_deg=12.0,
        sector_half_width_deg=5.0,
        config=INCMConfig(),
    )

    assert abs(result.target_angle_deg - 10.0) <= 2.0
    assert np.min(np.abs(result.selected_interference_angles_deg + 34.0)) <= 2.0
    assert np.isclose(np.vdot(result.weights, target), 1.0, atol=0.08)

    delay_sum = steering_dictionary(geometry, np.asarray([12.0]), 2_000.0)[:, 0]
    delay_sum /= np.vdot(delay_sum, delay_sum).real
    sbl_interference_gain = abs(np.vdot(result.weights, interferer))
    delay_sum_interference_gain = abs(np.vdot(delay_sum, interferer))
    assert sbl_interference_gain < 0.2 * delay_sum_interference_gain


def test_wideband_sbl_mvdr_returns_finite_waveform() -> None:
    sample_rate = 8_000
    geometry = CabinGeometry(microphone_count=4, microphone_spacing_m=0.04)
    rng = np.random.default_rng(23)
    audio = rng.normal(scale=0.1, size=(4, 1_024))
    result = wideband_sbl_mvdr(
        audio,
        sample_rate,
        geometry,
        visual_target_angle_deg=20.0,
        visual_confidence=0.8,
        sbl_config=SBLConfig(max_iterations=3),
        wideband_config=WidebandConfig(
            n_fft=128,
            hop_length=64,
            max_frequency_hz=3_000.0,
            frequency_stride=8,
            angle_step_deg=15.0,
        ),
    )
    assert result.enhanced_audio.shape == (audio.shape[1],)
    assert np.isfinite(result.enhanced_audio).all()
    assert result.processed_bins.size > 0
    assert 0.0 <= result.sbl_converged_fraction <= 1.0
    assert 0.0 < result.processed_bin_fraction <= 1.0
