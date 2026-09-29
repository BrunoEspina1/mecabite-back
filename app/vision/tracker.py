"""Adaptador de MediaPipe Hand Landmarker para cámara, imagen y video."""

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
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = ROOT_DIR / "models" / "vision" / "hand_landmarker.task"


def ensure_model() -> Path:
    """Descarga el modelo de MediaPipe solo la primera vez."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(".part")
    print("Descargando el modelo de MediaPipe (~8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
    except urllib.error.URLError:
        subprocess.run(["curl", "-fsSL", "-o", str(partial), MODEL_URL], check=True)
    partial.rename(MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    """Devuelve los 21 puntos de una mano por frame."""

    def __init__(self, images: bool = False, min_confidence: float = 0.5):
        self._images = images
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE if images else vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

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
            return None
        points = np.array([[point.x, point.y, point.z] for point in result.hand_landmarks[0]])
        is_left = result.handedness[0][0].category_name == "Left"
        return points, is_left

    def close(self) -> None:
        self._landmarker.close()


def draw_hand(frame: np.ndarray, points: np.ndarray) -> None:
    height, width = frame.shape[:2]
    pixels = [(int(x * width), int(y * height)) for x, y, _ in points]
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
