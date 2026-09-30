"""Correcciones concretas (RF-13): qué dedo estirar, hacia dónde girar la mano, subir el brazo.

Junta el guante y la cámara contra `expected` de la seña objetivo (signs.json):

- Configuración: el guante dice cuánto se flexiona cada dedo (1 encogido, 2 a medias,
  3 estirado). Donde el guante no sirve (la C marca todo en 3; el pulgar marca 2 muy seguido)
  se usan los ángulos de los dedos que da MediaPipe.
- Orientación: la cámara mide si la palma se ve de frente o de lado y hacia dónde apuntan los
  dedos. El guante (MPU) compara su inclinación con la de referencia de la seña
  (glove_reference.json); el giro (`yaw`) se deriva con el tiempo y no se usa.
- Brazo: con el cuerpo, si la mano está demasiado abajo. Es solo una sugerencia: no rechaza.

Los puntos vienen en las unidades de HandTracker/BodyTracker (alturas de imagen, vista en
espejo, y hacia abajo).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from statistics import median

import numpy as np

from app.catalog import Expected, Sign
from app.core.config import settings
from app.vision.body import LEFT_SHOULDER, PALM, RIGHT_SHOULDER, _visible
from app.vision.glove import FINGERS, GloveReading

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "catalog" / "glove_reference.json"
FINGER_NAMES = {
    "pulgar": "el pulgar",
    "indice": "el dedo índice",
    "medio": "el dedo medio",
    "anular": "el dedo anular",
    "menique": "el meñique",
}
# Con 3 dedos o más para el mismo lado se da una sola indicación para toda la mano.
WHOLE_HAND = 3

# Cámara: suma de lo que se doblan las 3 articulaciones de un dedo (radianes; 0 = recto).
EXTENDED_MAX_BEND = 1.0
FLEXED_MIN_BEND = 2.8
# Pulgar: distancia de la punta al nudillo del medio, en largos de palma.
THUMB_EXTENDED_MIN = 0.85
THUMB_FLEXED_MAX = 0.55
FINGER_JOINTS = {
    "indice": (0, 5, 6, 7, 8),
    "medio": (0, 9, 10, 11, 12),
    "anular": (0, 13, 14, 15, 16),
    "menique": (0, 17, 18, 19, 20),
}
# Ancho de los nudillos (índice a meñique) respecto al largo de la palma: de frente ~0.8,
# de lado cerca de 0.
FACING_MIN_WIDTH = 0.55
SIDE_MAX_WIDTH = 0.4
POINTING_MIN = 0.5  # componente vertical de muñeca -> nudillo del medio para "arriba"/"abajo"
LOW_HAND = 1.1  # palma más abajo que esto (anchos de hombro bajo los hombros): subir la mano


@dataclass(frozen=True)
class Correction:
    component: str  # configuration | orientation | localization
    part: str  # un dedo, "hand", "palm", "fingers", "wrist" o "arm"
    action: str  # extend | flex | curve | open | rotate_facing | rotate_side | point_up | ...
    message: str
    source: str  # glove | camera
    hint: bool = False  # solo sugerencia: no hace fallar el intento

    @property
    def key(self) -> tuple[str, str]:
        return (self.part, self.action)

    def to_json(self) -> dict:
        data = asdict(self)
        del data["hint"]
        return data


@dataclass(frozen=True)
class GloveState:
    """Lo que dice el guante en este momento (mediana de las lecturas recientes)."""

    fingers: dict[str, float]
    roll: float
    pitch: float

    @classmethod
    def from_readings(cls, readings: list[GloveReading]) -> GloveState | None:
        if not readings:
            return None
        return cls(
            fingers={f: median(r.fingers[f] for r in readings) for f in FINGERS},
            roll=median(r.roll for r in readings),
            pitch=median(r.pitch for r in readings),
        )

    def to_json(self) -> dict:
        return {
            "fingers": {finger: round(value, 1) for finger, value in self.fingers.items()},
            "roll": round(self.roll, 1),
            "pitch": round(self.pitch, 1),
        }


@cache
def glove_reference(path: Path = REFERENCE_PATH) -> dict[str, dict]:
    """Inclinación del guante con cada seña bien hecha (`vision glove-reference`)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("signs", {})


def evaluate(
    sign: Sign,
    hand: np.ndarray | None,
    body: np.ndarray | None,
    glove: GloveState | None,
    reference: dict | None = None,
) -> list[Correction]:
    """Todas las correcciones para parecerse a `sign`, de la más a la menos importante.

    `hand`: 21 puntos de la mano quieta (None si se mueve o no se ve: no se revisa con cámara).
    """
    expected = sign.expected
    corrections = []
    if glove is not None:
        corrections += _glove_fingers(expected, glove)
    if hand is not None:
        corrections += _camera_fingers(expected, hand)
        corrections += _camera_orientation(expected, hand)
    if glove is not None and reference:
        corrections += _glove_tilt(glove, reference)
    if hand is not None and body is not None:
        corrections += _arm_height(hand, body)
    return corrections


# --- Configuración ------------------------------------------------------------------------


def _glove_fingers(expected: Expected, glove: GloveState) -> list[Correction]:
    wrong = []
    for finger in FINGERS:
        allowed = expected.glove_fingers.get(finger)
        if not allowed:
            continue
        value = glove.fingers[finger]
        if value < min(allowed) - 0.25:
            wrong.append((finger, "extend"))
        elif value > max(allowed) + 0.25:
            wrong.append((finger, "flex"))
    return _finger_corrections(wrong, "glove")


def _finger_corrections(wrong: list[tuple[str, str]], source: str) -> list[Correction]:
    verbs = {
        "extend": "Estira más",
        "flex": "Encoge más",
        "curve": "Curva más",
        "open": "Abre un poco",
    }
    whole = {
        "extend": "Abre la mano: estira más los dedos",
        "flex": "Cierra más la mano: encoge los dedos",
        "curve": "Curva más los dedos",
        "open": "Abre un poco los dedos, sin estirarlos",
    }
    corrections = []
    for action in verbs:
        fingers = [finger for finger, wanted in wrong if wanted == action]
        if len(fingers) >= WHOLE_HAND:
            corrections.append(Correction("configuration", "hand", action, whole[action], source))
            continue
        corrections += [
            Correction(
                "configuration", finger, action, f"{verbs[action]} {FINGER_NAMES[finger]}", source
            )
            for finger in fingers
        ]
    order = {finger: index for index, finger in enumerate((*FINGERS, "hand"))}
    return sorted(corrections, key=lambda c: order[c.part])


def _bend(points: np.ndarray, chain: tuple[int, ...]) -> float:
    total = 0.0
    for before, joint, after in zip(chain, chain[1:], chain[2:], strict=False):
        first, second = points[before] - points[joint], points[after] - points[joint]
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator == 0:
            continue
        angle = np.arccos(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        total += np.pi - angle
    return float(total)


def camera_finger_states(points: np.ndarray) -> dict[str, str]:
    """extended | half | flexed para cada dedo, con los ángulos de MediaPipe."""
    states = {}
    for finger, chain in FINGER_JOINTS.items():
        bend = _bend(points, chain)
        states[finger] = (
            "extended"
            if bend <= EXTENDED_MAX_BEND
            else "flexed"
            if bend >= FLEXED_MIN_BEND
            else "half"
        )
    palm = np.linalg.norm(points[9] - points[0]) or 1.0
    reach = np.linalg.norm(points[4] - points[9]) / palm
    states["pulgar"] = (
        "extended"
        if reach >= THUMB_EXTENDED_MIN
        else "flexed"
        if reach <= THUMB_FLEXED_MAX
        else "half"
    )
    return states


def _camera_fingers(expected: Expected, points: np.ndarray) -> list[Correction]:
    if not expected.camera_fingers:
        return []
    states = camera_finger_states(points)
    wrong = []
    for finger, wanted in expected.camera_fingers.items():
        seen = states[finger]
        if seen == wanted:
            continue
        if finger == "pulgar" and seen == "half":
            continue  # la medida del pulgar es gruesa: solo se corrige el extremo contrario
        if wanted == "half":
            wrong.append((finger, "curve" if seen == "extended" else "open"))
        else:
            wrong.append((finger, "extend" if wanted == "extended" else "flex"))
    return _finger_corrections(wrong, "camera")


# --- Orientación --------------------------------------------------------------------------


def _camera_orientation(expected: Expected, points: np.ndarray) -> list[Correction]:
    corrections = []
    palm_length = np.linalg.norm(points[9, :2] - points[0, :2])
    if palm_length == 0:
        return []
    width = np.linalg.norm(points[5, :2] - points[17, :2]) / palm_length
    if expected.palm == "facing" and width < SIDE_MAX_WIDTH:
        corrections.append(
            Correction(
                "orientation",
                "palm",
                "rotate_facing",
                "Gira la palma hacia la cámara",
                "camera",
            )
        )
    elif expected.palm == "side" and width > FACING_MIN_WIDTH:
        corrections.append(
            Correction(
                "orientation",
                "palm",
                "rotate_side",
                "Gira la mano de lado: la palma mira hacia un costado",
                "camera",
            )
        )

    direction = (points[9, :2] - points[0, :2]) / palm_length  # y hacia abajo
    if expected.pointing == "up" and direction[1] > -POINTING_MIN:
        corrections.append(
            Correction(
                "orientation", "fingers", "point_up", "Apunta los dedos hacia arriba", "camera"
            )
        )
    elif expected.pointing == "down" and direction[1] < POINTING_MIN:
        corrections.append(
            Correction(
                "orientation", "fingers", "point_down", "Apunta los dedos hacia abajo", "camera"
            )
        )
    return corrections


def _glove_tilt(glove: GloveState, reference: dict) -> list[Correction]:
    tolerance = settings.glove_tilt_tolerance_deg
    corrections = []
    pitch = glove.pitch - reference["pitch"]
    if pitch > tolerance:
        corrections.append(
            Correction("orientation", "wrist", "tilt_down", "Inclina la mano hacia abajo", "glove")
        )
    elif pitch < -tolerance:
        corrections.append(
            Correction("orientation", "wrist", "tilt_up", "Inclina la mano hacia arriba", "glove")
        )
    roll = glove.roll - reference["roll"]
    if roll > tolerance:
        corrections.append(
            Correction(
                "orientation", "wrist", "roll_left", "Gira la muñeca hacia la izquierda", "glove"
            )
        )
    elif roll < -tolerance:
        corrections.append(
            Correction(
                "orientation", "wrist", "roll_right", "Gira la muñeca hacia la derecha", "glove"
            )
        )
    return corrections


# --- Brazo --------------------------------------------------------------------------------


def _arm_height(points: np.ndarray, body: np.ndarray) -> list[Correction]:
    if not _visible(body, LEFT_SHOULDER, RIGHT_SHOULDER):
        return []
    left, right = body[LEFT_SHOULDER, :2], body[RIGHT_SHOULDER, :2]
    width = float(np.linalg.norm(left - right))
    if width <= 0:
        return []
    below = (points[PALM, 1].mean() - (left[1] + right[1]) / 2) / width
    if below <= LOW_HAND:
        return []
    return [
        Correction(
            "localization",
            "arm",
            "raise",
            "Sube más el brazo: la mano a la altura del pecho",
            "camera",
            hint=True,
        )
    ]


# --- Estabilidad --------------------------------------------------------------------------


class CorrectionFilter:
    """Solo muestra una corrección que se mantiene `hold_ms`.

    Las lecturas cambian cuadro a cuadro (el guante a ~30 Hz, MediaPipe tiembla): sin esto los
    mensajes parpadean. Una ausencia menor a `grace_ms` no la reinicia.
    """

    def __init__(self, hold_ms: float = 400, grace_ms: float = 250):
        self.hold_ms = hold_ms
        self.grace_ms = grace_ms
        self._seen: dict[tuple[str, str], tuple[float, float, Correction]] = {}

    def update(self, corrections: list[Correction], t_ms: float) -> list[Correction]:
        for correction in corrections:
            first = self._seen.get(correction.key, (t_ms, t_ms, correction))[0]
            self._seen[correction.key] = (first, t_ms, correction)
        self._seen = {
            key: seen for key, seen in self._seen.items() if t_ms - seen[1] <= self.grace_ms
        }
        order = {correction.key: index for index, correction in enumerate(corrections)}
        stable = [
            correction
            for first, _last, correction in self._seen.values()
            if t_ms - first >= self.hold_ms
        ]
        return sorted(stable, key=lambda c: order.get(c.key, len(order)))

    def reset(self) -> None:
        self._seen.clear()
