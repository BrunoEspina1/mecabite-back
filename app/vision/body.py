"""Cuerpo como referencia: encuadre frente a la cámara y localización de la mano (RF-03).

Todo se mide en "anchos de hombro" desde el centro de los hombros, así no depende de qué
tan cerca esté la persona. Los puntos vienen de BodyTracker/HandTracker (unidades de la
altura de la imagen, x hacia la derecha, y hacia abajo).

Los mensajes van sin acentos: las fuentes de OpenCV solo tienen ASCII.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 2, 5, 7, 8
MOUTH_LEFT, MOUTH_RIGHT, LEFT_SHOULDER, RIGHT_SHOULDER = 9, 10, 11, 12
PALM = [0, 5, 9, 13, 17]  # muñeca y nudillos de la mano: centro de la palma

MIN_VISIBILITY = 0.5
MIN_SHOULDER_WIDTH = 0.18  # en alturas de imagen; menos: la persona está muy lejos
CHEST_ROOM = 0.5  # bajo los hombros debe caber medio ancho de hombros (ahí se hacen las letras)
CENTER_MARGIN = 0.2  # el centro de los hombros debe quedar entre el 20 % y el 80 % del ancho
MAX_NOSE_OFFSET = 0.3  # nariz respecto al centro de los hombros: más, está girada
MAX_SHOULDER_DEPTH = 0.8  # diferencia de profundidad entre hombros: más, está de lado


@dataclass(frozen=True)
class BodyStatus:
    issue: str | None  # qué corregir para quedar bien frente a la cámara; None si está bien
    location: str | None  # zona del cuerpo donde está la mano; None si no se puede medir

    @property
    def framed(self) -> bool:
        return self.issue is None


def _visible(body: np.ndarray, *indices: int) -> bool:
    return bool(np.all(body[list(indices), 3] >= MIN_VISIBILITY))


def _shoulders(body: np.ndarray) -> tuple[np.ndarray, float]:
    left, right = body[LEFT_SHOULDER, :2], body[RIGHT_SHOULDER, :2]
    return (left + right) / 2, float(np.linalg.norm(left - right))


def framing_issue(body: np.ndarray | None, aspect: float) -> str | None:
    """`aspect` = ancho / alto de la imagen (el ancho en las unidades de los puntos)."""
    if body is None:
        return "No se ve tu cuerpo: ponte frente a la camara"
    if not _visible(body, NOSE):
        return "No se ve tu cara"
    shoulders_x = body[[LEFT_SHOULDER, RIGHT_SHOULDER], 0]
    inside = np.all((shoulders_x >= 0) & (shoulders_x <= aspect))
    if not _visible(body, LEFT_SHOULDER, RIGHT_SHOULDER) or not inside:
        return "Alejate un poco: no se ven tus dos hombros"
    center, width = _shoulders(body)
    if width < MIN_SHOULDER_WIDTH:
        return "Acercate a la camara"
    if center[1] + CHEST_ROOM * width > 1.0:
        return "Alejate un poco: no se ve tu pecho"
    if not CENTER_MARGIN * aspect <= center[0] <= (1 - CENTER_MARGIN) * aspect:
        return "Muevete al centro de la imagen"
    turned = abs(body[NOSE, 0] - center[0]) / width > MAX_NOSE_OFFSET
    sideways = abs(body[LEFT_SHOULDER, 2] - body[RIGHT_SHOULDER, 2]) / width > MAX_SHOULDER_DEPTH
    if turned or sideways:
        return "Ponte de frente a la camara"
    return None


def hand_location(hand: np.ndarray | None, body: np.ndarray | None) -> str | None:
    """Zona del cuerpo en la que está el centro de la palma."""
    if hand is None or body is None or not _visible(body, NOSE, LEFT_SHOULDER, RIGHT_SHOULDER):
        return None
    center, width = _shoulders(body)
    if width <= 0:
        return None

    def relative(point: np.ndarray) -> np.ndarray:
        return (point[:2] - center) / width

    palm = relative(hand[PALM, :2].mean(axis=0))
    nose = relative(body[NOSE])
    eyes_y = relative(body[[LEFT_EYE, RIGHT_EYE], :2].mean(axis=0))[1]
    mouth_y = relative(body[[MOUTH_LEFT, MOUTH_RIGHT], :2].mean(axis=0))[1]
    face_half = 0.2
    if _visible(body, LEFT_EAR, RIGHT_EAR):
        face_half = max(abs(body[LEFT_EAR, 0] - body[RIGHT_EAR, 0]) / width / 2, face_half)
    from_face = abs(palm[0] - nose[0])

    if palm[1] <= mouth_y + 0.3:  # a la altura de la cabeza
        if from_face <= face_half * 1.2:
            if palm[1] < eyes_y - 0.05:
                return "frente"
            if palm[1] < mouth_y - 0.08:
                return "cara"
            return "boca/barbilla"
        if from_face <= face_half * 2.8:
            return "lado de la cara"
        return "arriba, lejos del cuerpo"
    if palm[1] <= 0.25 and 0.3 <= abs(palm[0]) <= 0.9:
        return "hombro"
    if abs(palm[0]) <= 0.8 and palm[1] <= 1.3:
        return "pecho"
    return "lejos del cuerpo"


def body_status(hand: np.ndarray | None, body: np.ndarray | None, aspect: float) -> BodyStatus:
    return BodyStatus(framing_issue(body, aspect), hand_location(hand, body))
