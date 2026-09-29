"""Convierte landmarks de MediaPipe en características de la mano."""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
TIPS = [4, 8, 12, 16, 20]
THUMB_TARGETS = [5, 6, 9, 10, 13, 14, 17]
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]
NUM_FEATURES = NUM_LANDMARKS * 3 + 10 + len(THUMB_TARGETS) + 3 * len(FINGER_CHAINS) + 1


def _normalize(landmarks: np.ndarray, is_left: bool) -> np.ndarray:
    points = landmarks.astype(np.float64).copy()
    if is_left:
        points[:, 0] = -points[:, 0]
    points -= points[WRIST]
    scale = np.linalg.norm(points[MIDDLE_MCP])
    if scale > 0:
        points /= scale
    return points


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first, second = a - b, c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    cosine = np.dot(first, second) / denominator
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def landmarks_to_vector(landmarks: np.ndarray, is_left: bool = False) -> np.ndarray:
    """Convierte 21 landmarks (x, y, z) en un vector normalizado de 96 features."""
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"Se esperaban ({NUM_LANDMARKS}, 3) landmarks, no {landmarks.shape}")

    points = _normalize(landmarks, is_left)
    tip_distances = [
        np.linalg.norm(points[a] - points[b]) for i, a in enumerate(TIPS) for b in TIPS[i + 1 :]
    ]
    thumb_distances = [np.linalg.norm(points[4] - points[target]) for target in THUMB_TARGETS]
    angles = [
        _angle(points[chain[index - 1]], points[chain[index]], points[chain[index + 1]])
        for chain in FINGER_CHAINS
        for index in (1, 2, 3)
    ]
    knuckles = points[5, :2] - points[9, :2]
    tips = points[8, :2] - points[12, :2]
    crossed = float(np.dot(tips, knuckles) / (np.dot(knuckles, knuckles) or 1.0))

    vector = np.concatenate([points.flatten(), tip_distances, thumb_distances, angles, [crossed]])
    if vector.size != NUM_FEATURES:
        raise RuntimeError(
            f"El vector de visión tiene {vector.size} features; se esperaban {NUM_FEATURES}"
        )
    return vector
