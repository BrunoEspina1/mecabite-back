"""Adaptadores de MediaPipe (mano y cuerpo) para cámara, imagen y video."""

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

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


class HandTracker:
    """Devuelve los 21 puntos de una mano por frame.

    Los puntos salen en unidades de la altura de la imagen (x y z se escalan por ancho/alto):
    MediaPipe normaliza x al ancho e y al alto, así que en una webcam 16:9 la mano se vería
    aplastada frente a los datasets cuadrados.

    `mirrored`: el frame llega volteado como espejo (webcam y videos importados). MediaPipe
    nombra la mano según cómo se ve en la imagen, así que en espejo la etiqueta se invierte.
    """

    def __init__(self, images: bool = False, min_confidence: float = 0.5, mirrored: bool = True):
        self._images = images
        self._mirrored = mirrored
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE if images else vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.handedness_score = 0.0

    def detect(
        self, frame_bgr: np.ndarray, timestamp_ms: int = 0
    ) -> tuple[np.ndarray, bool] | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = (
            self._landmarker.detect(image)
            if self._images
            else self._landmarker.detect_for_video(image, timestamp_ms)
        )
        if not result.hand_landmarks:
            self.handedness_score = 0.0
            return None
        height, width = frame_bgr.shape[:2]
        points = np.array([[point.x, point.y, point.z] for point in result.hand_landmarks[0]])
        points[:, [0, 2]] *= width / height
        # Con puño o mano de perfil el score baja y la etiqueta brinca.
        category = result.handedness[0][0]
        self.handedness_score = float(category.score)
        return points, category.category_name == ("Right" if self._mirrored else "Left")

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


def draw_hand(frame: np.ndarray, points: np.ndarray) -> None:
    height = frame.shape[0]
    pixels = [(int(x * height), int(y * height)) for x, y, _ in points]
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pixels[connection.start], pixels[connection.end], (80, 220, 120), 2)
    for point in pixels:
        cv2.circle(frame, point, 4, (255, 255, 255), -1)


def open_camera(index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara {index}. Revisa el permiso de cámara de la terminal."
        )
    return capture
