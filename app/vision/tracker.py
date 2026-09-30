"""Adaptadores de MediaPipe (mano y cuerpo) para cámara, imagen y video."""

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import TypeVar

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from app.vision.features import Hand

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models" / "vision"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
# Versión lite: la misma que correría en el teléfono (RNF-02, 20 fps en gama media).
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
NUM_BODY_LANDMARKS = 33
MAX_HAND_JUMP = 0.25  # alturas de imagen: de un cuadro al siguiente una mano no salta más
FORGET_HAND_MS = 1000  # sin ver una mano este tiempo, se olvida dónde estaba
PRIMARY_LOST_MS = 300  # sin la principal este tiempo, la otra pasa a ser la principal
# Mano dominante: las muñecas deben estar al menos así de separadas (alturas de imagen) para
# saber cuál es cuál por su posición; si no, se usan las etiquetas de MediaPipe.
MIN_WRIST_SEPARATION = 0.05
LABEL_MIN_SCORE = 0.8  # etiqueta de MediaPipe confiable para decidir qué mano es
SWAP_AFTER_FRAMES = 5  # cuadros seguidos que la otra debe parecer la dominante para cambiarlas
# Puntos del torso hacia arriba (cara, hombros, brazos, cadera); las piernas no se dibujan.
UPPER_BODY = 25


def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Descarga un modelo de MediaPipe solo la primera vez."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    print(f"Descargando el modelo de MediaPipe {path.name}...")
    try:
        urllib.request.urlretrieve(url, partial)
    except urllib.error.URLError:
        subprocess.run(["curl", "-fsSL", "-o", str(partial), url], check=True)
    partial.rename(path)
    return path


H = TypeVar("H", bound=tuple)  # una mano: (puntos, ...)


class HandSlots:
    """Dos lugares fijos para las manos: MediaPipe no las entrega siempre en el mismo orden.

    Cada mano va al lugar donde estaba su muñeca antes. El lugar 0 es la mano principal: la
    primera que apareció, como la seguiría MediaPipe con una sola mano. Con
    `promote_after_ms`, si la principal no se ve ese tiempo y la otra sí, la otra pasa a ser
    la principal.

    Con `dominant_left` (la mano con la que la persona hace las señas) y las dos manos a la
    vista, la principal es la dominante: se decide al verlas juntas por primera vez y después
    se sigue por continuidad; solo se cambian si la otra parece la dominante
    SWAP_AFTER_FRAMES cuadros seguidos, y nunca con `locked` (a media trayectoria). Las manos
    son tuplas (puntos, izquierda, score) como las de `HandTracker.detect_hands`.
    """

    def __init__(self, promote_after_ms: float | None = None, dominant_left: bool | None = None):
        self.promote_after_ms = promote_after_ms
        self.dominant_left = dominant_left
        self._last: list[tuple[float, np.ndarray] | None] = [None, None]  # (t_ms, muñeca)
        self._seen = [False, False]  # si el lugar tuvo mano en el cuadro anterior
        self._primary_ms = 0.0
        self._both_ms: float | None = None  # última vez que se vieron las dos manos
        self._swap_votes = 0

    def dominant_slot(self, first: H, second: H) -> int | None:
        """0 o 1: cuál de las dos es la dominante; None si no se puede saber.

        En las unidades de HandTracker (vista en espejo) la mano derecha de la persona queda
        con la muñeca más a la derecha. Si las muñecas están casi alineadas (manos juntas o
        cruzadas) se usan las etiquetas de MediaPipe, solo si ambas son confiables y distintas.
        """
        first_x, second_x = first[0][0, 0], second[0][0, 0]
        if abs(first_x - second_x) >= MIN_WRIST_SEPARATION:
            right = 0 if first_x > second_x else 1
        elif first[1] != second[1] and min(first[2], second[2]) >= LABEL_MIN_SCORE:
            right = 1 if first[1] else 0
        else:
            return None
        return 1 - right if self.dominant_left else right

    def assign(self, hands: list[H], t_ms: float, locked: bool = False) -> list[H | None]:
        self._last = [
            last if last is not None and t_ms - last[0] <= FORGET_HAND_MS else None
            for last in self._last
        ]

        def distance(hand: H, slot: int) -> float:
            last = self._last[slot]
            return 0.0 if last is None else float(np.linalg.norm(hand[0][0, :2] - last[1]))

        slots: list[H | None] = [None, None]
        if len(hands) >= 2:
            crossed = distance(hands[1], 0) + distance(hands[0], 1)
            straight = distance(hands[0], 0) + distance(hands[1], 1)
            slots = [hands[1], hands[0]] if crossed < straight else [hands[0], hands[1]]
        elif hands:
            known = [slot for slot in (0, 1) if self._last[slot] is not None]
            slot = min(known, key=lambda slot: distance(hands[0], slot)) if known else 0
            # Tras perderla unos cuadros sí puede reaparecer lejos (entró a cuadro y subió).
            if self._seen[slot] and distance(hands[0], slot) > MAX_HAND_JUMP:
                slot = 1 - slot
            slots[slot] = hands[0]

        if self.dominant_left is not None:
            slots = self._put_dominant_first(slots, t_ms, locked)
        if slots[0] is not None:
            self._primary_ms = t_ms
        elif (
            slots[1] is not None
            and self.promote_after_ms is not None
            and t_ms - self._primary_ms > self.promote_after_ms
        ):
            slots.reverse()
            self._last.reverse()
            self._primary_ms = t_ms
        for slot, hand in enumerate(slots):
            if hand is not None:
                self._last[slot] = (t_ms, hand[0][0, :2])
        self._seen = [hand is not None for hand in slots]
        return slots

    def _put_dominant_first(self, slots: list[H | None], t_ms: float, locked: bool) -> list:
        if slots[0] is None or slots[1] is None:
            if self._both_ms is not None and t_ms - self._both_ms > FORGET_HAND_MS:
                self._both_ms = None  # al volver a verlas juntas se decide otra vez
            return slots
        dominant = self.dominant_slot(slots[0], slots[1])
        first_time = self._both_ms is None
        self._both_ms = t_ms
        if dominant == 1:
            self._swap_votes += 1
        elif dominant == 0:
            self._swap_votes = 0
        if dominant == 1 and not locked and (first_time or self._swap_votes >= SWAP_AFTER_FRAMES):
            slots = [slots[1], slots[0]]
            self._last.reverse()
            self._seen.reverse()
            self._swap_votes = 0
        return slots


def apply_sides(
    primary: tuple | None, other: tuple | None, dominant_left: bool | None
) -> tuple[tuple | None, tuple | None, bool]:
    """Pone a cada mano el lado que eligió la persona, no la etiqueta de MediaPipe.

    MediaPipe se equivoca de lado con el puño o de perfil, y el lado decide si los puntos se
    comparan en espejo con los modelos. Devuelve (principal, otra, solo_la_otra): con una sola
    mano que MediaPipe asegura que es la no dominante, se deja su lado y `solo_la_otra` es True.
    """
    if dominant_left is None or primary is None:
        return primary, other, False
    points, is_left, score = primary[:3]
    if other is None and is_left != dominant_left and score >= LABEL_MIN_SCORE:
        return primary, other, True
    primary = (points, dominant_left, score)
    if other is not None:
        other = (other[0], not dominant_left, other[2])
    return primary, other, False


HAND_NAMES = {"derecha": False, "right": False, "izquierda": True, "left": True}


class HandTracker:
    """Devuelve los 21 puntos de la mano principal por frame; la segunda mano queda en `other`.

    Los puntos salen en unidades de la altura de la imagen (x y z se escalan por ancho/alto):
    MediaPipe normaliza x al ancho e y al alto, así que en una webcam 16:9 la mano se vería
    aplastada frente a los datasets cuadrados.

    `mirrored`: el frame llega volteado como espejo (webcam y videos importados). MediaPipe
    nombra la mano según cómo se ve en la imagen, así que en espejo la etiqueta se invierte.

    Se detectan hasta dos manos (señas como gracias usan ambas). La principal es la misma que
    seguiría MediaPipe con una sola mano: la primera que apareció, mientras siga a la vista
    (`HandSlots`). Las letras se reconocen solo con ella. Con `dominant_left` (`--hand`) la
    principal es la mano con la que la persona hace las señas y cada mano toma su lado.
    """

    def __init__(
        self,
        images: bool = False,
        min_confidence: float = 0.5,
        mirrored: bool = True,
        dominant_left: bool | None = None,
    ):
        self._images = images
        self._mirrored = mirrored
        self.dominant_left = dominant_left
        self.only_other_hand = False
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE if images else vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.handedness_score = 0.0
        self.other: Hand | None = None
        self._slots = HandSlots(promote_after_ms=PRIMARY_LOST_MS, dominant_left=dominant_left)

    def detect(
        self, frame_bgr: np.ndarray, timestamp_ms: int = 0, locked: bool = False
    ) -> Hand | None:
        """Mano principal (o None); la otra, si se ve, queda en `self.other`.

        Si la principal se pierde un momento, devuelve None aunque la otra se vea: así la
        seña de una mano no brinca a la otra a media trayectoria. `locked` (a media
        trayectoria): no se cambia cuál es la dominante."""
        hands = self.detect_hands(frame_bgr, timestamp_ms)
        primary, other = self._slots.assign(hands, timestamp_ms, locked)
        primary, other, self.only_other_hand = apply_sides(primary, other, self.dominant_left)
        self.other = other[:2] if other else None
        self.handedness_score = primary[2] if primary else 0.0
        return primary[:2] if primary else None

    def detect_hands(
        self, frame_bgr: np.ndarray, timestamp_ms: int = 0
    ) -> list[tuple[np.ndarray, bool, float]]:
        """Todas las manos del cuadro, en el orden de MediaPipe: (puntos, izquierda, score)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = (
            self._landmarker.detect(image)
            if self._images
            else self._landmarker.detect_for_video(image, timestamp_ms)
        )
        height, width = frame_bgr.shape[:2]
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness, strict=True):
            points = np.array([[point.x, point.y, point.z] for point in landmarks])
            points[:, [0, 2]] *= width / height
            # Con puño o mano de perfil el score baja y la etiqueta brinca.
            category = handedness[0]
            is_left = category.category_name == ("Right" if self._mirrored else "Left")
            hands.append((points, is_left, float(category.score)))
        return hands

    def close(self) -> None:
        self._landmarker.close()


class BodyTracker:
    """33 puntos del cuerpo (cara, hombros, brazos...) en las mismas unidades que HandTracker.

    Devuelve (33, 4): x, y, z y la visibilidad (0-1) de cada punto. Con ellos se mide dónde
    está la mano respecto a la cara y los hombros (localización, RF-03) y si la persona está
    de frente a la cámara.
    """

    def __init__(self, min_confidence: float = 0.5):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL))
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        height, width = frame_bgr.shape[:2]
        points = np.array(
            [[point.x, point.y, point.z, point.visibility] for point in result.pose_landmarks[0]]
        )
        points[:, [0, 2]] *= width / height
        return points

    def close(self) -> None:
        self._landmarker.close()


def draw_body(frame: np.ndarray, body: np.ndarray, min_visibility: float = 0.5) -> None:
    height = frame.shape[0]
    pixels = [(int(x * height), int(y * height)) for x, y, _, _ in body]
    visible = body[:, 3] >= min_visibility
    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        start, end = connection.start, connection.end
        if start < UPPER_BODY and end < UPPER_BODY and visible[start] and visible[end]:
            cv2.line(frame, pixels[start], pixels[end], (255, 170, 60), 2)
    for index in range(UPPER_BODY):
        if visible[index]:
            cv2.circle(frame, pixels[index], 4, (255, 170, 60), -1)


def draw_hand(frame: np.ndarray, points: np.ndarray, color=(80, 220, 120)) -> None:
    height = frame.shape[0]
    pixels = [(int(x * height), int(y * height)) for x, y, _ in points]
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pixels[connection.start], pixels[connection.end], color, 2)
    for point in pixels:
        cv2.circle(frame, point, 4, (255, 255, 255), -1)


def open_camera(index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara {index}. Revisa el permiso de cámara de la terminal."
        )
    return capture
