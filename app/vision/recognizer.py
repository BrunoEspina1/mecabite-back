"""Reconocimiento en vivo a partir de landmarks, sin cámara ni ventanas.

Lo usan `vision demo/live/practice` con la webcam y lo usará la API con los datos que manda
el iPhone. Todo avanza con el `t_ms` de cada observación (no con el reloj de la máquina), así
da igual si los cuadros llegan a 30 fps desde la webcam o a 15 Hz por la red.

Estáticas: el modelo estático vota cuadro por cuadro durante VOTE_MS y la seña ganadora se
confirma al sostenerla HOLD_SECONDS (RF-06). Con movimiento: el modelo dinámico la confirma al
terminar la trayectoria (RF-07).

Las letras se reconocen con la mano principal. Las señas de dos manos del catálogo (gracias,
por favor) se siguen en cualquiera de las dos, porque en algunas una mano queda quieta de
base, y solo cuentan si la otra mano también participa.
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.vision.catalog import OTHER, load_catalog
from app.vision.classifier import SignClassifier
from app.vision.features import Hand, landmarks_to_vector
from app.vision.sequence import DynamicDetector

ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_MODEL_PATH = ROOT_DIR / "models" / "vision" / "static.joblib"
DYNAMIC_MODEL_PATH = ROOT_DIR / "models" / "vision" / "dynamic.joblib"
# Ajustables desde .env (VISION_*): ver .env.example.
CONFIDENCE_THRESHOLD = settings.vision_confidence_threshold
VOTE_MS = settings.vision_vote_ms  # ventana de votos del modelo estático
VOTE_SHARE = 0.75  # la seña debe ganar este porcentaje de la ventana
HOLD_SECONDS = settings.vision_hold_ms / 1000
GRACE_SECONDS = settings.vision_grace_ms / 1000
HANDEDNESS_MIN_SCORE = 0.8
HAND_VOTES = 12  # cuadros para decidir si es mano izquierda o derecha (solo informativo)
ALPHABET = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"


def alphabet_key(sign: str) -> tuple[int, str]:
    # La Ñ va después de la N (el orden de Python la pondría al final).
    position = ALPHABET.find(sign)
    return (position if position >= 0 else len(ALPHABET), sign)


class SignStabilizer:
    """Confirma una seña solo cuando se mantiene `hold_seconds` seguidos.

    Evita que las formas intermedias de un movimiento (por ejemplo, la I al inicio de la J)
    aparezcan como seña reconocida.

    Un cambio que dura menos de `grace_seconds` (la mano se movió un poco) no reinicia el
    tiempo sostenido. Si el cambio sigue, la seña nueva cuenta desde que apareció.
    """

    def __init__(self, hold_seconds: float = HOLD_SECONDS, grace_seconds: float = GRACE_SECONDS):
        self.hold_seconds = hold_seconds
        self.grace_seconds = grace_seconds
        self.candidate: str | None = None
        self.since = 0.0
        self.confirmed: str | None = None
        self._change: tuple[str | None, float] | None = None  # (seña distinta, desde)

    def update(self, label: str | None, now: float) -> str | None:
        """Devuelve la seña en el instante en que se confirma; si no, None."""
        if label == self.candidate:
            self._change = None
        else:
            if self._change is None or self._change[0] != label:
                self._change = (label, now)
            if now - self._change[1] < self.grace_seconds:
                return None
            self.candidate, self.since = self._change
            self._change = None
        if now - self.since < self.hold_seconds or label == self.confirmed:
            return None
        self.confirmed = label
        return label

    def force(self, label: str, now: float, show_seconds: float = 2.0) -> None:
        """Muestra una seña confirmada por otro camino (con movimiento) al menos `show_seconds`."""
        self.confirmed, self.candidate, self._change = label, None, None
        self.since = now + show_seconds - self.hold_seconds

    def progress(self, now: float) -> float:
        if self.candidate is None or self.candidate == self.confirmed:
            return 0.0
        return min(max((now - self.since) / self.hold_seconds, 0.0), 1.0)


class Recognizer:
    """Estado del reconocimiento de una persona; se alimenta cuadro por cuadro con `update`.

    Las clases son las de los modelos (nombres de carpeta: `A`, `Ñ`, `K`); traducirlas a ids
    del catálogo de la app (`app.catalog`) le toca a quien lo use.
    """

    def __init__(
        self,
        classifier: SignClassifier,
        detector: DynamicDetector | None = None,
        second_detector: DynamicDetector | None = None,
    ):
        self.classifier = classifier
        self.detector = detector  # mano principal: todas las señas con movimiento
        self.second_detector = second_detector  # segunda mano: solo señas de dos manos
        self.reset()

    @classmethod
    def load(
        cls, static_path: Path = STATIC_MODEL_PATH, dynamic_path: Path = DYNAMIC_MODEL_PATH
    ) -> Recognizer:
        if not static_path.exists():
            raise FileNotFoundError("No hay modelo. Ejecuta primero: vision train")
        dynamic = None
        if dynamic_path.exists():
            dynamic = SignClassifier.load(dynamic_path)
        else:
            print("Aviso: sin modelo dinámico, solo se reconocen señas estáticas.")
        return cls.from_classifiers(SignClassifier.load(static_path), dynamic)

    @classmethod
    def from_classifiers(
        cls, static: SignClassifier, dynamic: SignClassifier | None = None
    ) -> Recognizer:
        """Un reconocedor con el estado en cero sobre modelos ya cargados.

        Los clasificadores se pueden compartir (la API los carga una vez); los detectores
        guardan la trayectoria de una persona, así que cada reconocedor tiene los suyos.
        """
        detector = second_detector = None
        if dynamic is not None:
            two_handed = load_catalog().two_handed & set(dynamic.labels)
            detector = DynamicDetector(dynamic, OTHER, two_handed)
            if two_handed:
                second_detector = DynamicDetector(dynamic, OTHER, two_handed, two_handed_only=True)
        return cls(static, detector, second_detector)

    @property
    def signs(self) -> list[str]:
        """Señas que los modelos cargados saben reconocer, en orden alfabético."""
        labels = set(self.classifier.labels)
        if self.detector is not None:
            labels |= set(self.detector.classifier.labels)
        labels.discard(OTHER)
        return sorted(labels, key=alphabet_key)

    def is_dynamic(self, sign: str) -> bool:
        return self.detector is not None and sign in self.detector.classifier.labels

    @property
    def is_left(self) -> bool | None:
        """Mano con la que se está haciendo la seña; None si aún no hay votos confiables."""
        if not self.hands:
            return None
        return sum(self.hands) > len(self.hands) / 2

    @property
    def moving(self) -> bool:
        """Alguna mano está a media trayectoria."""
        return any(d is not None and d.moving for d in (self.detector, self.second_detector))

    @property
    def moving_detector(self) -> DynamicDetector | None:
        """Detector de la última seña con movimiento confirmada (sus cuadros son esa seña)."""
        return self._moved or self.detector

    def reset(self) -> None:
        self.votes: deque[tuple[float, str | None]] = deque()
        self.hands: deque[bool] = deque(maxlen=HAND_VOTES)
        self.stabilizer = SignStabilizer()
        self.frame_label: tuple[str, float] | None = None
        self.rejected: str | None = None
        self.last_movement: tuple[str, float] | None = None  # última trayectoria clasificada
        self._moved: DynamicDetector | None = None
        for detector in (self.detector, self.second_detector):
            if detector is not None:
                detector.reset()

    def update(
        self,
        detection: Hand | None,
        handedness_score: float,
        t_ms: float,
        other: Hand | None = None,
        body: np.ndarray | None = None,
        max_movement_duration_ms: int | None = None,
    ) -> str | None:
        """Procesa un cuadro. Devuelve la seña en el instante en que se confirma.

        `detection`: (21 puntos, mano izquierda) de la mano principal como los entrega
        `HandTracker` (unidades de la altura de la imagen, vista en espejo), o None si no hay
        mano. `handedness_score`: score de MediaPipe para la etiqueta de mano. `t_ms`: tiempo
        del cuadro, creciente. `other`: la segunda mano y `body`: los 33 puntos del cuerpo, si
        se ven; sin ellos, las señas de dos manos no se aceptan.

        Si en este cuadro se descartó una seña de dos manos hecha con una, queda en `rejected`.
        """
        now = t_ms / 1000
        if detection:
            points, is_left = detection
            label, confidence = self.classifier.predict(landmarks_to_vector(points, is_left))
            self.frame_label = (label, confidence)
            accepted = label != OTHER and confidence >= CONFIDENCE_THRESHOLD
            self.votes.append((t_ms, label if accepted else None))
            if handedness_score >= HANDEDNESS_MIN_SCORE:
                self.hands.append(is_left)
        else:
            self.frame_label = None
            self.votes.append((t_ms, None))
            self.hands.clear()
        while t_ms - self.votes[0][0] > VOTE_MS:
            self.votes.popleft()

        new_sign = self.stabilizer.update(self._voted(), now)
        self.rejected = None
        slots = ((self.detector, detection, other), (self.second_detector, other, detection))
        for detector, hand, partner in slots:
            if detector is None:
                continue
            points, is_left = hand if hand else (None, False)
            before = detector.last
            moving_sign = detector.push(
                t_ms, points, is_left, partner, body, max_movement_duration_ms
            )
            if detector.last is not before:
                self.last_movement = detector.last
            self.rejected = self.rejected or detector.rejected
            if moving_sign is not None:
                self.stabilizer.force(moving_sign, now)
                self.votes.clear()
                self._moved = detector
                # Si las dos manos se movían (por favor), la otra no la vuelve a contar.
                for rest in (self.detector, self.second_detector):
                    if rest is not None and rest is not detector:
                        rest.reset()
                return moving_sign
        return new_sign

    def progress(self, t_ms: float) -> float:
        """Avance (0 a 1) de la seña estática que se está sosteniendo."""
        return self.stabilizer.progress(t_ms / 1000)

    def _voted(self) -> str | None:
        """Seña que gana VOTE_SHARE de los votos, si ya hay casi una ventana completa."""
        if not self.votes or self.votes[-1][0] - self.votes[0][0] < VOTE_MS * VOTE_SHARE:
            return None
        label, count = Counter(label for _, label in self.votes).most_common(1)[0]
        return label if count >= len(self.votes) * VOTE_SHARE else None
