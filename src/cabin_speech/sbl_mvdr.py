from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from cabin_speech.geometry import ArrayGeometry


@dataclass(frozen=True)
class SBLConfig:
    """MMV-SBL settings matching Table 2 and Appendix A of the paper."""

    init_noise_ratio: float = 0.1
    gamma_floor: float = 1e-12
    noise_floor: float = 1e-12
    tolerance: float = 1e-4
    max_iterations: int = 200


@dataclass(frozen=True)
class INCMConfig:
    """Leakage-aware INCM reconstruction settings from the paper."""

    threshold_alpha: float = 10.0
    coverage_ratio: float = 0.9
    diagonal_loading_beta: float = 1e-3


@dataclass(frozen=True)
class WidebandConfig:
    """Speech-domain settings layered on top of the narrowband paper method."""

    n_fft: int = 512
    hop_length: int = 128
    min_frequency_hz: float = 200.0
    max_frequency_hz: float = 6_000.0
    frequency_stride: int = 1
    angle_min_deg: float = -90.0
    angle_max_deg: float = 90.0
    angle_step_deg: float = 3.0
    default_sector_half_width_deg: float = 5.0
    min_sector_half_width_deg: float = 3.0
    max_sector_half_width_deg: float = 15.0


@dataclass
class SBLResult:
    posterior_mean: np.ndarray
    gamma: np.ndarray
    noise_variance: float
    power_spectrum: np.ndarray
    iterations: int
    converged: bool


@dataclass
class INCMResult:
    weights: np.ndarray
    target_angle_deg: float
    selected_interference_angles_deg: np.ndarray
    interference_covariance: np.ndarray
    loaded_incm: np.ndarray


@dataclass
class WidebandResult:
    enhanced_audio: np.ndarray
    frequencies_hz: np.ndarray
    target_angles_deg: np.ndarray
    processed_bins: np.ndarray
    sector_half_width_deg: float
    mean_sbl_iterations: float
    sbl_converged_fraction: float
    processed_bin_fraction: float


def steering_dictionary(
    geometry: ArrayGeometry,
    angles_deg: np.ndarray,
    frequency_hz: float,
) -> np.ndarray:
    """Construct the ULA narrowband steering dictionary A in Eq. (20)."""

    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("angles_deg must be a non-empty one-dimensional array")
    if frequency_hz < 0:
        raise ValueError("frequency_hz must be non-negative")

    delays = np.column_stack(
        [geometry.propagation_delays_s(float(angle_deg)) for angle_deg in angles]
    )
    return np.exp(-2j * np.pi * float(frequency_hz) * delays)


def run_mmv_sbl(
    snapshots: np.ndarray,
    dictionary: np.ndarray,
    config: SBLConfig | None = None,
) -> SBLResult:
    """Estimate a row-sparse angular spectrum using Appendix A MMV-SBL.

    ``snapshots`` has shape ``[microphones, snapshots]`` and ``dictionary``
    has shape ``[microphones, angular_grid_points]``.
    """

    cfg = config or SBLConfig()
    x = np.asarray(snapshots, dtype=np.complex128)
    a = np.asarray(dictionary, dtype=np.complex128)
    if x.ndim != 2 or a.ndim != 2:
        raise ValueError("snapshots and dictionary must both be two-dimensional")
    if x.shape[0] != a.shape[0]:
        raise ValueError("snapshot and dictionary microphone dimensions must match")
    if x.shape[1] == 0:
        raise ValueError("at least one snapshot is required")
    if cfg.max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    microphone_count, snapshot_count = x.shape
    covariance = x @ x.conj().T / snapshot_count
    covariance = (covariance + covariance.conj().T) / 2.0
    average_sensor_power = max(
        float(np.trace(covariance).real / microphone_count),
        cfg.noise_floor,
    )
    noise_variance = max(cfg.init_noise_ratio * average_sensor_power, cfg.noise_floor)

    projected_power = np.real(np.diag(a.conj().T @ covariance @ a))
    gamma = np.maximum(projected_power, cfg.gamma_floor)
    identity = np.eye(microphone_count, dtype=np.complex128)
    posterior_mean = np.zeros((a.shape[1], snapshot_count), dtype=np.complex128)
    converged = False

    for _iteration in range(1, cfg.max_iterations + 1):
        model_covariance = noise_variance * identity + (a * gamma[np.newaxis, :]) @ a.conj().T
        model_covariance = (model_covariance + model_covariance.conj().T) / 2.0

        solved_x = np.linalg.solve(model_covariance, x)
        solved_a = np.linalg.solve(model_covariance, a)
        posterior_mean = gamma[:, np.newaxis] * (a.conj().T @ solved_x)

        posterior_diagonal = gamma - np.square(gamma) * np.real(np.sum(a.conj() * solved_a, axis=0))
        posterior_diagonal = np.maximum(posterior_diagonal, 0.0)
        gamma_new = np.mean(np.abs(posterior_mean) ** 2, axis=1) + posterior_diagonal
        gamma_new = np.maximum(gamma_new, cfg.gamma_floor)

        residual = x - a @ posterior_mean
        noise_new = float(
            np.linalg.norm(residual, ord="fro") ** 2 / (microphone_count * snapshot_count)
        )
        noise_new = max(noise_new, cfg.noise_floor)

        gamma_change = np.linalg.norm(gamma_new - gamma) / max(
            np.linalg.norm(gamma),
            cfg.gamma_floor,
        )
        noise_change = abs(noise_new - noise_variance) / max(
            noise_variance,
            cfg.noise_floor,
        )
        gamma = gamma_new
        noise_variance = noise_new
        if gamma_change < cfg.tolerance and noise_change < cfg.tolerance:
            converged = True
            break

    # Recompute the posterior using the final hyperparameters.
    model_covariance = noise_variance * identity + (a * gamma[np.newaxis, :]) @ a.conj().T
    posterior_mean = gamma[:, np.newaxis] * (a.conj().T @ np.linalg.solve(model_covariance, x))
    power_spectrum = np.mean(np.abs(posterior_mean) ** 2, axis=1)
    return SBLResult(
        posterior_mean=posterior_mean,
        gamma=gamma,
        noise_variance=noise_variance,
        power_spectrum=power_spectrum,
        iterations=_iteration,
        converged=converged,
    )


def adaptive_sector_half_width(
    visual_confidence: float | None,
    config: WidebandConfig | None = None,
) -> float:
    """Map visual confidence to a protected target sector.

    This confidence-adaptive rule is the project extension, not part of the
    original paper. High confidence narrows the sector and low confidence
    protects a wider region from being mistaken for interference.
    """

    cfg = config or WidebandConfig()
    if visual_confidence is None:
        return cfg.default_sector_half_width_deg
    confidence = float(np.clip(visual_confidence, 0.0, 1.0))
    return cfg.max_sector_half_width_deg - confidence * (
        cfg.max_sector_half_width_deg - cfg.min_sector_half_width_deg
    )


def _local_maxima(power: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    candidates: list[int] = []
    for index in np.flatnonzero(valid_mask):
        left = max(index - 1, 0)
        right = min(index + 1, power.size - 1)
        if power[index] >= power[left] and power[index] >= power[right]:
            candidates.append(int(index))
    return np.asarray(candidates, dtype=np.int64)


def reconstruct_incm_mvdr(
    power_spectrum: np.ndarray,
    noise_variance: float,
    dictionary: np.ndarray,
    angles_deg: np.ndarray,
    nominal_target_angle_deg: float,
    sector_half_width_deg: float = 5.0,
    config: INCMConfig | None = None,
) -> INCMResult:
    """Apply Eqs. (31), (34), and (39)-(44) of the SBL-INCM paper."""

    cfg = config or INCMConfig()
    power = np.asarray(power_spectrum, dtype=np.float64)
    a = np.asarray(dictionary, dtype=np.complex128)
    angles = np.asarray(angles_deg, dtype=np.float64)
    if power.ndim != 1 or angles.shape != power.shape:
        raise ValueError("power_spectrum and angles_deg must have the same 1-D shape")
    if a.shape[1] != power.size:
        raise ValueError("dictionary grid dimension does not match power_spectrum")
    if not 0.0 < cfg.coverage_ratio <= 1.0:
        raise ValueError("coverage_ratio must be in (0, 1]")
    if noise_variance < 0:
        raise ValueError("noise_variance must be non-negative")

    desired_mask = np.abs(angles - float(nominal_target_angle_deg)) <= sector_half_width_deg
    if not np.any(desired_mask):
        nearest = int(np.argmin(np.abs(angles - nominal_target_angle_deg)))
        desired_mask[nearest] = True
    desired_indices = np.flatnonzero(desired_mask)
    target_index = int(desired_indices[np.argmax(power[desired_indices])])
    target_steering = a[:, target_index]

    interference_mask = ~desired_mask
    candidates = _local_maxima(power, interference_mask)
    threshold = cfg.threshold_alpha * float(noise_variance)
    screened = candidates[power[candidates] > threshold]
    if screened.size:
        screened = screened[np.argsort(power[screened])[::-1]]
        out_of_sector_power = float(np.sum(power[interference_mask]))
        required_power = cfg.coverage_ratio * out_of_sector_power
        cumulative = np.cumsum(power[screened])
        reaching = np.flatnonzero(cumulative >= required_power)
        selected_count = int(reaching[0] + 1) if reaching.size else screened.size
        selected = screened[:selected_count]
    else:
        selected = np.empty(0, dtype=np.int64)

    excess_power = np.maximum(power - float(noise_variance), 0.0)
    total_scale = float(np.sum(excess_power[interference_mask]))
    microphone_count = a.shape[0]
    interference_covariance = np.zeros(
        (microphone_count, microphone_count),
        dtype=np.complex128,
    )
    if selected.size and total_scale > 0.0:
        selected_excess = excess_power[selected]
        selected_weights = selected_excess / max(
            float(np.sum(selected_excess)),
            np.finfo(float).eps,
        )
        direction_covariance = np.zeros_like(interference_covariance)
        for index, weight in zip(selected, selected_weights, strict=True):
            atom = a[:, index]
            direction_covariance += weight * np.outer(atom, atom.conj())
        interference_covariance = total_scale * direction_covariance

    identity = np.eye(microphone_count, dtype=np.complex128)
    incm = interference_covariance + float(noise_variance) * identity
    loading = cfg.diagonal_loading_beta * float(np.trace(incm).real) / microphone_count
    loaded_incm = incm + loading * identity
    inverse_target = np.linalg.solve(loaded_incm, target_steering)
    denominator = np.vdot(target_steering, inverse_target)
    weights = inverse_target / max(float(denominator.real), np.finfo(float).eps)

    return INCMResult(
        weights=weights,
        target_angle_deg=float(angles[target_index]),
        selected_interference_angles_deg=angles[selected],
        interference_covariance=interference_covariance,
        loaded_incm=loaded_incm,
    )


def _delay_and_sum_weights(
    geometry: ArrayGeometry,
    target_angle_deg: float,
    frequency_hz: float,
) -> np.ndarray:
    steering = steering_dictionary(
        geometry,
        np.asarray([target_angle_deg]),
        frequency_hz,
    )[:, 0]
    return steering / np.vdot(steering, steering).real


def wideband_sbl_mvdr(
    multichannel_audio: np.ndarray,
    sample_rate: int,
    geometry: ArrayGeometry,
    visual_target_angle_deg: float,
    visual_confidence: float | None = None,
    sbl_config: SBLConfig | None = None,
    incm_config: INCMConfig | None = None,
    wideband_config: WidebandConfig | None = None,
) -> WidebandResult:
    """Enhance wideband speech with visual-prior-guided per-bin SBL-INCM MVDR."""

    audio = np.asarray(multichannel_audio, dtype=np.float64)
    cfg = wideband_config or WidebandConfig()
    if audio.ndim != 2:
        raise ValueError("multichannel_audio must have shape [channels, samples]")
    if audio.shape[0] != geometry.microphone_count:
        raise ValueError("channel count does not match geometry")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if cfg.frequency_stride < 1:
        raise ValueError("frequency_stride must be at least one")
    if cfg.n_fft > audio.shape[1]:
        raise ValueError("audio must contain at least n_fft samples")

    noverlap = cfg.n_fft - cfg.hop_length
    frequencies, _, spectra = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=cfg.n_fft,
        noverlap=noverlap,
        nfft=cfg.n_fft,
        axis=-1,
        boundary="zeros",
        padded=True,
    )
    enhanced_spectrum = np.zeros_like(spectra[0])
    angles = np.arange(
        cfg.angle_min_deg,
        cfg.angle_max_deg + 0.5 * cfg.angle_step_deg,
        cfg.angle_step_deg,
    )
    half_width = adaptive_sector_half_width(visual_confidence, cfg)
    eligible = np.flatnonzero(
        (frequencies >= cfg.min_frequency_hz)
        & (frequencies <= min(cfg.max_frequency_hz, sample_rate / 2.0))
    )
    processed = eligible[:: cfg.frequency_stride]
    if processed.size == 0:
        raise ValueError("no STFT bins fall inside the configured frequency range")

    target_angles: list[float] = []
    iterations: list[int] = []
    convergence: list[bool] = []
    computed_weights: dict[int, np.ndarray] = {}
    for bin_index in processed:
        dictionary = steering_dictionary(geometry, angles, frequencies[bin_index])
        sbl_result = run_mmv_sbl(spectra[:, bin_index, :], dictionary, sbl_config)
        incm_result = reconstruct_incm_mvdr(
            power_spectrum=sbl_result.power_spectrum,
            noise_variance=sbl_result.noise_variance,
            dictionary=dictionary,
            angles_deg=angles,
            nominal_target_angle_deg=visual_target_angle_deg,
            sector_half_width_deg=half_width,
            config=incm_config,
        )
        computed_weights[int(bin_index)] = incm_result.weights
        target_angles.append(incm_result.target_angle_deg)
        iterations.append(sbl_result.iterations)
        convergence.append(sbl_result.converged)

    for bin_index, frequency in enumerate(frequencies):
        if bin_index in computed_weights:
            weights = computed_weights[bin_index]
        else:
            weights = _delay_and_sum_weights(
                geometry,
                visual_target_angle_deg,
                frequency,
            )
        enhanced_spectrum[bin_index] = np.conj(weights) @ spectra[:, bin_index, :]

    _, enhanced = signal.istft(
        enhanced_spectrum,
        fs=sample_rate,
        window="hann",
        nperseg=cfg.n_fft,
        noverlap=noverlap,
        nfft=cfg.n_fft,
        input_onesided=True,
        boundary=True,
    )
    enhanced = np.asarray(enhanced[: audio.shape[1]], dtype=np.float64)
    if enhanced.size < audio.shape[1]:
        enhanced = np.pad(enhanced, (0, audio.shape[1] - enhanced.size))

    return WidebandResult(
        enhanced_audio=enhanced,
        frequencies_hz=frequencies[processed],
        target_angles_deg=np.asarray(target_angles),
        processed_bins=processed,
        sector_half_width_deg=half_width,
        mean_sbl_iterations=float(np.mean(iterations)),
        sbl_converged_fraction=float(np.mean(convergence)),
        processed_bin_fraction=float(processed.size / frequencies.size),
    )
