"""Letras con movimiento (J, Ñ, Q, X, Z) a partir de la secuencia de la mano.

La secuencia se remuestrea por tiempo a RATE_HZ (da igual si la cámara va a 30 o 60 fps) y el
modelo ve los últimos WINDOW_SECONDS a velocidad real, igual al entrenar que en vivo. Así se
compara la trayectoria completa dentro de una ventana menor a 3 s (RF-07).

Features por ventana: forma de la mano al inicio/medio/final, cuánto cambia, y la trayectoria
de la muñeca y de la punta del índice medidas en "tamaños de palma".
"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path

import numpy as np

from app.vision.features import MIDDLE_MCP, NUM_FEATURES, NUM_LANDMARKS, WRIST, landmarks_to_vector

RATE_HZ = 15
WINDOW_SECONDS = 2.0  # los videos del dataset duran 1.6-2.5 s
WINDOW = round(RATE_HZ * WINDOW_SECONDS)
TRAJECTORY_POINTS = 5
INDEX_TIP = 8
MAX_GAP_MS = 300  # si la mano se pierde más que esto, la secuencia empieza de nuevo
CONFIDENCE_THRESHOLD = 0.6
# Velocidad media de los 21 puntos en palmas/s. En los videos del dataset el pico de una seña
# supera ~5 y la mano en reposo queda por debajo de ~0.8; hay histéresis entre ambos umbrales.
MOVING_SPEED = 2.5
STILL_SPEED = 1.2
SPEED_LAG_MS = (
    150  # se mide contra el cuadro de hace 150 ms para no confundir temblor con movimiento
)
END_STILL_MS = 250  # la seña terminó cuando la mano lleva este tiempo quieta


def load_sequence(path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """CSV de `collect` o `import-videos` -> (t_ms (T,), puntos (T, 21, 3), mano izquierda)."""
    with path.open(newline="") as file_handle:
        reader = csv.reader(file_handle)
        next(reader, None)
        rows = [row for row in reader if row]
    # Las fotos (MSL-ABC) no tienen tiempo: la columna viene vacía.
    t_ms = np.array([float(row[0]) if row[0] else np.nan for row in rows])
    is_left = np.array([row[1] == "1" for row in rows])
    # Después de la mano puede haber columnas del cuerpo (collect/practice): aquí no se usan.
    hand_columns = slice(2, 2 + NUM_LANDMARKS * 3)
    points = np.asarray([row[hand_columns] for row in rows], dtype=float)
    points = points.reshape(-1, NUM_LANDMARKS, 3)
    return t_ms, points, bool(is_left.mean() > 0.5) if len(rows) else False


def resample(t_ms: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Interpola los puntos a tiempos uniformes (RATE_HZ por segundo)."""
    if len(t_ms) < 2:
        return points.copy()
    count = int((t_ms[-1] - t_ms[0]) * RATE_HZ / 1000) + 1
    grid = t_ms[0] + np.arange(count) * 1000 / RATE_HZ
    flat = points.reshape(len(points), -1)
    columns = [np.interp(grid, t_ms, flat[:, column]) for column in range(flat.shape[1])]
    return np.stack(columns, axis=1).reshape(count, NUM_LANDMARKS, 3)


def frame_vectors(points_seq: np.ndarray, is_left: bool) -> np.ndarray:
    """(N, 21, 3) -> (N, 96 + 5): forma, muñeca (x, y), punta del índice (x, y), palma.

    Con `is_left` se refleja toda la secuencia (forma y trayectoria): una letra hecha con la
    izquierda es el espejo de la misma letra con la derecha.
    """
    rows = []
    for points in points_seq:
        mirrored = points.copy()
        if is_left:
            mirrored[:, 0] = -mirrored[:, 0]
        palm = max(float(np.linalg.norm(mirrored[MIDDLE_MCP, :2] - mirrored[WRIST, :2])), 1e-6)
        tip = (mirrored[INDEX_TIP, :2] - mirrored[WRIST, :2]) / palm
        shape = landmarks_to_vector(mirrored, is_left=False)
        rows.append(np.concatenate([shape, mirrored[WRIST, :2], tip, [palm]]))
    return np.asarray(rows)


def last_window(vectors: np.ndarray) -> np.ndarray:
    """Últimos WINDOW cuadros; si faltan, se repite el primero al inicio (mano quieta)."""
    if len(vectors) < WINDOW:
        padding = np.repeat(vectors[:1], WINDOW - len(vectors), axis=0)
        vectors = np.concatenate([padding, vectors])
    return vectors[-WINDOW:]


def window_features(window: np.ndarray) -> np.ndarray:
    shape = window[:, :NUM_FEATURES]
    wrist = window[:, NUM_FEATURES : NUM_FEATURES + 2]
    tip = window[:, NUM_FEATURES + 2 : NUM_FEATURES + 4]
    palm = max(float(window[:, NUM_FEATURES + 4].mean()), 1e-6)
    trajectory = (wrist - wrist[0]) / palm
    speed = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    finger_speed = np.linalg.norm(np.diff(shape[:, : NUM_LANDMARKS * 3], axis=0), axis=1)
    samples = np.linspace(0, len(window) - 1, TRAJECTORY_POINTS).round().astype(int)
    return np.concatenate(
        [
            *(third.mean(axis=0) for third in np.array_split(shape, 3)),
            shape.std(axis=0),
            trajectory[samples].ravel(),
            trajectory.std(axis=0),
            tip[samples].ravel(),
            tip.std(axis=0),
            [speed.sum(), speed.max(), finger_speed.sum(), finger_speed.max()],
        ]
    )


def sequence_features(t_ms: np.ndarray, points: np.ndarray, is_left: bool) -> np.ndarray:
    return window_features(last_window(frame_vectors(resample(t_ms, points), is_left)))


# --- Datos de entrenamiento ------------------------------------------------------------------


def augment(
    t_ms: np.ndarray, points: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Otra velocidad, pausas antes/después y temblor: nadie hace la seña igual que el video."""
    t_ms = (t_ms - t_ms[0]) * rng.uniform(0.75, 1.3)
    lead, tail = rng.uniform(1, 400), rng.uniform(1, 500)
    t_ms = np.concatenate([[0.0], t_ms + lead, [t_ms[-1] + lead + tail]])
    points = np.concatenate([points[:1], points, points[-1:]])
    return t_ms, points + rng.normal(0, 0.002, points.shape)


def held_pose(points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Una forma quieta con temblor y deriva lenta: negativo (la forma de la X quieta no es X)."""
    count = int(rng.uniform(1.0, 2.5) * RATE_HZ)
    drift = np.cumsum(rng.normal(0, 0.0015, (count, 1, 2)), axis=0)
    sequence = np.repeat(points[None], count, axis=0) + rng.normal(0, 0.002, (count, *points.shape))
    sequence[:, :, :2] += drift
    return np.arange(count) * 1000 / RATE_HZ, sequence


def transition(
    start: np.ndarray, end: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Pasar de una forma a otra (con algo de desplazamiento): negativo."""
    palm_start = np.linalg.norm(start[MIDDLE_MCP, :2] - start[WRIST, :2])
    palm_end = max(np.linalg.norm(end[MIDDLE_MCP, :2] - end[WRIST, :2]), 1e-6)
    offset = np.zeros(3)
    offset[:2] = rng.normal(0, 0.4 * palm_start, 2)
    end = (end - end[WRIST]) * (palm_start / palm_end) + start[WRIST] + offset
    hold_a, move, hold_b = (
        int(rng.uniform(*span) * 30) for span in ((0.2, 0.8), (0.3, 0.8), (0.2, 0.8))
    )
    weights = np.concatenate([np.zeros(hold_a), np.linspace(0, 1, max(move, 2)), np.ones(hold_b)])
    sequence = start[None] + weights[:, None, None] * (end - start)[None]
    sequence += rng.normal(0, 0.002, sequence.shape)
    return np.arange(len(sequence)) * 1000 / 30, sequence


# --- En vivo ---------------------------------------------------------------------------------


def hand_speed(earlier: np.ndarray, later: np.ndarray, elapsed_ms: float) -> float:
    """Desplazamiento medio de los 21 puntos, en palmas por segundo."""
    palm = max(float(np.linalg.norm(later[MIDDLE_MCP, :2] - later[WRIST, :2])), 1e-6)
    displacement = np.linalg.norm(later[:, :2] - earlier[:, :2], axis=1).mean()
    return float(displacement / palm / max(elapsed_ms / 1000, 1e-3))


class DynamicDetector:
    """Clasifica cada movimiento una sola vez, cuando termina.

    Un movimiento empieza cuando la mano supera MOVING_SPEED y termina cuando lleva
    END_STILL_MS por debajo de STILL_SPEED (o al durar WINDOW_SECONDS). Entonces se compara la
    trayectoria completa: así no se confirma una X a media Q ni se cuenta dos veces la misma
    seña.
    """

    def __init__(self, classifier, other_label: str):
        self.classifier = classifier
        self.other_label = other_label
        self._buffer: deque[tuple[float, np.ndarray, bool]] = deque()
        self._moving_since: float | None = None
        self._still_since: float | None = None
        self.speed = 0.0
        self.last: tuple[str, float] | None = None

    @property
    def moving(self) -> bool:
        return self._moving_since is not None

    def push(self, t_ms: float, points: np.ndarray | None, is_left: bool = False) -> str | None:
        """Agrega un cuadro (`points=None` si no hay mano). Devuelve la letra al confirmarse."""
        if points is None:
            if self._buffer and t_ms - self._buffer[-1][0] > MAX_GAP_MS:
                self.reset()
            return None
        self._buffer.append((t_ms, points, is_left))
        while t_ms - self._buffer[0][0] > WINDOW_SECONDS * 1000 + 500:
            self._buffer.popleft()

        self.speed = self._current_speed(t_ms, points)
        if self.speed >= MOVING_SPEED:
            if self._moving_since is None:
                self._moving_since = t_ms
            self._still_since = None
        elif self.speed < STILL_SPEED and self._still_since is None:
            self._still_since = t_ms

        if self._moving_since is None:
            return None
        ended = self._still_since is not None and t_ms - self._still_since >= END_STILL_MS
        too_long = t_ms - self._moving_since >= WINDOW_SECONDS * 1000
        if not (ended or too_long):
            return None
        self._moving_since = None
        return self._classify()

    def _current_speed(self, t_ms: float, points: np.ndarray) -> float:
        for earlier_ms, earlier, _ in reversed(self._buffer):
            if t_ms - earlier_ms >= SPEED_LAG_MS:
                return hand_speed(earlier, points, t_ms - earlier_ms)
        return 0.0

    def _classify(self) -> str | None:
        times = np.array([frame[0] for frame in self._buffer])
        sequence = np.array([frame[1] for frame in self._buffer])
        left = np.mean([frame[2] for frame in self._buffer]) > 0.5
        label, confidence = self.classifier.predict(sequence_features(times, sequence, left))
        self.last = (label, confidence)
        if label == self.other_label or confidence < CONFIDENCE_THRESHOLD:
            return None
        return label

    def frames(self) -> list[tuple[float, np.ndarray, bool]]:
        """Cuadros con mano acumulados (t_ms, puntos, mano izquierda): la última seña completa."""
        return list(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._moving_since = self._still_since = None
        self.last = None
