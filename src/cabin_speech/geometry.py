from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ArrayGeometry(Protocol):
    """Geometry interface shared by linear and planar microphone arrays."""

    microphone_count: int
    speed_of_sound_m_s: float

    def propagation_delays_s(self, angle_deg: float) -> np.ndarray:
        """Return one far-field propagation delay per microphone."""


@dataclass(frozen=True)
class CabinGeometry:
    """Simplified four-seat cabin and a centered uniform linear microphone array.

    Angles are measured from array broadside. Negative angles are on the left
    side of the cabin and positive angles are on the right side.
    """

    seat_angles_deg: tuple[float, ...] = (-55.0, -20.0, 20.0, 55.0)
    seat_names: tuple[str, ...] = (
        "driver",
        "front_passenger",
        "rear_left",
        "rear_right",
    )
    microphone_count: int = 4
    microphone_spacing_m: float = 0.05
    speed_of_sound_m_s: float = 343.0

    def __post_init__(self) -> None:
        if len(self.seat_angles_deg) != len(self.seat_names):
            raise ValueError("seat_angles_deg and seat_names must have equal length")
        if self.microphone_count < 2:
            raise ValueError("at least two microphones are required")
        if self.microphone_spacing_m <= 0:
            raise ValueError("microphone spacing must be positive")

    @property
    def num_seats(self) -> int:
        return len(self.seat_names)

    @property
    def microphone_positions_m(self) -> np.ndarray:
        indices = np.arange(self.microphone_count, dtype=np.float64)
        return (indices - indices.mean()) * self.microphone_spacing_m

    @property
    def microphone_positions_xy_m(self) -> np.ndarray:
        """Microphone coordinates with broadside along the positive y-axis."""

        return np.column_stack(
            (
                self.microphone_positions_m,
                np.zeros(self.microphone_count, dtype=np.float64),
            )
        )

    @property
    def array_aperture_m(self) -> float:
        positions = self.microphone_positions_m
        return float(positions[-1] - positions[0])

    def propagation_delays_s(self, angle_deg: float) -> np.ndarray:
        angle_rad = np.deg2rad(angle_deg)
        return self.microphone_positions_m * np.sin(angle_rad) / self.speed_of_sound_m_s

    def closest_seat(self, angle_deg: float) -> int:
        distances = np.abs(np.asarray(self.seat_angles_deg) - float(angle_deg))
        return int(np.argmin(distances))

    def angle_likelihoods(self, angle_deg: float, std_deg: float = 14.0) -> np.ndarray:
        if std_deg <= 0:
            raise ValueError("std_deg must be positive")
        distances = np.asarray(self.seat_angles_deg, dtype=np.float64) - float(angle_deg)
        scores = np.exp(-0.5 * np.square(distances / std_deg))
        total = scores.sum()
        return scores / total if total > 0 else np.full(self.num_seats, 1.0 / self.num_seats)


@dataclass(frozen=True)
class CircularArrayGeometry:
    """Uniform circular array in the horizontal plane.

    Angles are measured from the positive y-axis and increase toward the
    positive x-axis, matching ``CabinGeometry``. ``channel_zero_angle_deg``
    describes the physical azimuth of channel zero; changing it rotates only
    the reported DOA reference frame.
    """

    microphone_count: int = 8
    radius_m: float = 0.05
    channel_zero_angle_deg: float = 0.0
    speed_of_sound_m_s: float = 343.0

    def __post_init__(self) -> None:
        if self.microphone_count < 3:
            raise ValueError("a circular array requires at least three microphones")
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if self.speed_of_sound_m_s <= 0:
            raise ValueError("speed_of_sound_m_s must be positive")

    @property
    def microphone_angles_deg(self) -> np.ndarray:
        return self.channel_zero_angle_deg + np.arange(self.microphone_count) * (
            360.0 / self.microphone_count
        )

    @property
    def microphone_positions_xy_m(self) -> np.ndarray:
        microphone_angles_rad = np.deg2rad(self.microphone_angles_deg)
        return self.radius_m * np.column_stack(
            (
                np.sin(microphone_angles_rad),
                np.cos(microphone_angles_rad),
            )
        )

    def propagation_delays_s(self, angle_deg: float) -> np.ndarray:
        angle_rad = np.deg2rad(angle_deg)
        direction = np.asarray([np.sin(angle_rad), np.cos(angle_rad)])
        return self.microphone_positions_xy_m @ direction / self.speed_of_sound_m_s
