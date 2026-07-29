import numpy as np

from cabin_speech.vision import intersection_over_union, normalised_mouth_motion


def test_intersection_over_union() -> None:
    first = (10, 10, 40, 40)
    second = (20, 20, 40, 40)
    assert np.isclose(intersection_over_union(first, second), 900 / 2300)


def test_static_mouth_has_no_motion() -> None:
    image = np.full((32, 64), 120, dtype=np.uint8)
    assert normalised_mouth_motion(image, image.copy()) == 0.0


def test_changed_mouth_has_high_motion() -> None:
    previous = np.zeros((32, 64), dtype=np.uint8)
    current = np.full((32, 64), 80, dtype=np.uint8)
    assert normalised_mouth_motion(previous, current) > 0.9
