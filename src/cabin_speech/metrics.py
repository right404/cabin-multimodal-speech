from __future__ import annotations

from typing import Any

import numpy as np
from opencc import OpenCC

_TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")


def normalize_chinese_text(text: str) -> str:
    """Normalize ASR text for CER, including tags and traditional Chinese."""

    import re

    without_tags = re.sub(r"<[^>]*>", "", text)
    simplified = _TRADITIONAL_TO_SIMPLIFIED.convert(without_tags)
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", simplified)).lower()


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute character-level Levenshtein error rate."""

    reference_characters = list(reference)
    hypothesis_characters = list(hypothesis)
    if not reference_characters:
        return 0.0 if not hypothesis_characters else 1.0

    previous = list(range(len(hypothesis_characters) + 1))
    for reference_index, reference_character in enumerate(reference_characters, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_character in enumerate(
            hypothesis_characters,
            start=1,
        ):
            substitution_cost = int(reference_character != hypothesis_character)
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1] / len(reference_characters)


def scale_invariant_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Compute scale-invariant signal-to-distortion ratio in decibels."""

    target = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(estimate, dtype=np.float64)
    if target.ndim != 1 or prediction.ndim != 1:
        raise ValueError("reference and estimate must be one-dimensional")
    if target.shape != prediction.shape:
        raise ValueError("reference and estimate must have the same shape")
    target = target - np.mean(target)
    prediction = prediction - np.mean(prediction)
    target_energy = float(np.dot(target, target))
    if target_energy <= epsilon:
        raise ValueError("reference energy is too small")
    projection = np.dot(prediction, target) * target / target_energy
    residual = prediction - projection
    ratio = np.dot(projection, projection) / max(float(np.dot(residual, residual)), epsilon)
    return float(10.0 * np.log10(max(ratio, epsilon)))


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")

    class_count = len(class_names)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target, prediction] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
        f1_values.append(f1)
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
