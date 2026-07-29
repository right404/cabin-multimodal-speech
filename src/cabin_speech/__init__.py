"""Audio-visual target-speaker and array-processing research project."""

from __future__ import annotations

from typing import Any

from cabin_speech.geometry import CabinGeometry, CircularArrayGeometry

__all__ = [
    "CabinGeometry",
    "CircularArrayGeometry",
    "MultimodalFusionNet",
    "SyntheticCabinDataset",
]


def __getattr__(name: str) -> Any:
    """Lazily import PyTorch components so lightweight CLI commands stay usable."""

    if name == "MultimodalFusionNet":
        from cabin_speech.fusion import MultimodalFusionNet

        return MultimodalFusionNet
    if name == "SyntheticCabinDataset":
        from cabin_speech.synthetic import SyntheticCabinDataset

        return SyntheticCabinDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
