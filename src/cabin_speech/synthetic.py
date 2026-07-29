from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cabin_speech.geometry import CabinGeometry


@dataclass(frozen=True)
class SyntheticDataConfig:
    no_speech_probability: float = 0.12
    audio_missing_probability: float = 0.10
    visual_occlusion_probability: float = 0.18
    distractor_lip_probability: float = 0.15
    doa_noise_std_deg: float = 9.0

    def __post_init__(self) -> None:
        probabilities = (
            self.no_speech_probability,
            self.audio_missing_probability,
            self.visual_occlusion_probability,
            self.distractor_lip_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("all probabilities must be between zero and one")
        if self.doa_noise_std_deg <= 0:
            raise ValueError("doa_noise_std_deg must be positive")


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, None)
    total = values.sum()
    return values / total if total > 0 else np.zeros_like(values)


def generate_scenario(
    rng: np.random.Generator,
    geometry: CabinGeometry,
    config: SyntheticDataConfig,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Generate one audio-visual cabin observation.

    Feature layout:
      - 4 audio seat likelihoods
      - 4 lip-motion scores
      - 4 visual visibility scores
      - audio confidence
      - visual confidence
    """

    no_speech = rng.random() < config.no_speech_probability
    label = geometry.num_seats if no_speech else int(rng.integers(geometry.num_seats))

    audio_missing = rng.random() < config.audio_missing_probability
    if no_speech or audio_missing:
        observed_doa = float(rng.uniform(-75.0, 75.0))
        audio_confidence = float(rng.beta(1.3, 5.0))
        audio_scores = rng.uniform(0.0, 0.12, geometry.num_seats)
    else:
        true_angle = geometry.seat_angles_deg[label]
        observed_doa = float(rng.normal(true_angle, config.doa_noise_std_deg))
        audio_confidence = float(rng.beta(6.0, 1.8))
        audio_scores = geometry.angle_likelihoods(observed_doa)
        audio_scores += rng.normal(0.0, 0.025, geometry.num_seats)
    audio_scores = _normalise(audio_scores) * audio_confidence

    visibility = rng.uniform(0.72, 1.0, geometry.num_seats)
    occluded = rng.random(geometry.num_seats) < config.visual_occlusion_probability
    visibility[occluded] *= rng.uniform(0.02, 0.25, int(occluded.sum()))

    lip_scores = rng.beta(1.2, 8.0, geometry.num_seats) * visibility
    if not no_speech:
        lip_scores[label] = rng.beta(7.0, 1.5) * visibility[label]
    if rng.random() < config.distractor_lip_probability:
        candidates = [index for index in range(geometry.num_seats) if index != label]
        distractor = int(rng.choice(candidates))
        lip_scores[distractor] = max(lip_scores[distractor], rng.beta(5.0, 2.0))

    visual_confidence = float(np.max(visibility))
    features = np.concatenate(
        [
            audio_scores,
            lip_scores,
            visibility,
            np.asarray([audio_confidence, visual_confidence]),
        ]
    ).astype(np.float32)
    metadata = {
        "label": label,
        "observed_doa_deg": observed_doa,
        "audio_missing": audio_missing,
        "occluded_seats": np.flatnonzero(occluded).tolist(),
    }
    return features, label, metadata


class SyntheticCabinDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic synthetic dataset for fast, reproducible fusion experiments."""

    def __init__(
        self,
        size: int,
        geometry: CabinGeometry | None = None,
        config: SyntheticDataConfig | None = None,
        seed: int = 42,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.geometry = geometry or CabinGeometry()
        self.config = config or SyntheticDataConfig()
        self.size = int(size)

        rng = np.random.default_rng(seed)
        samples = [generate_scenario(rng, self.geometry, self.config)[:2] for _ in range(self.size)]
        self.features = torch.from_numpy(np.stack([sample[0] for sample in samples]))
        self.labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_classes(self) -> int:
        return self.geometry.num_seats + 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
