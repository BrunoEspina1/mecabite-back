"""Una sesión de práctica o demostración: observaciones de la app -> mensajes `feedback`.

Las reglas son las de `vision live --target` (`app/vision/teacher.py`), ahora sin cámara:
una ejecución de la seña objetivo cuenta solo si la persona está bien encuadrada (de frente,
con cara y hombros a la vista) y 3 correctas seguidas aprueban la seña (RF-10).

Los puntos llegan normalizados a la imagen vertical que vio MediaPipe en el iPhone. Se
convierten a las unidades de `HandTracker`/`BodyTracker` (vista en espejo, x y z en alturas
de imagen), las mismas con las que se entrenaron los modelos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.catalog import COMPONENTS, Catalog, Sign
from app.sessions.protocol import ObservationIn, VisionIn
from app.vision.body import BodyStatus, body_status
from app.vision.recognizer import Recognizer
from app.vision.tracker import PRIMARY_LOST_MS, HandSlots

APPROVE_AFTER = 3  # correctas seguidas (RF-10)
RESULT_SHOW_MS = 1500  # tiempo que se sigue reportando un resultado para que la app lo muestre
# body.py escribe sin acentos (las fuentes de OpenCV solo tienen ASCII); la app sí los tiene.
ACCENTS = {"camara": "cámara", "Alejate": "Aléjate", "Acercate": "Acércate", "Muevete": "Muévete"}


class ObservationError(ValueError):
    """Observación que no se puede procesar; la conexión sigue abierta."""


# Una mano como la entrega HandTracker.detect_hands: (puntos, izquierda, score).
TrackedHand = tuple[np.ndarray, bool, float]


def to_tracker_units(vision: VisionIn) -> tuple[list[TrackedHand], np.ndarray | None, float]:
    """Manos y cuerpo en las unidades de los trackers de la webcam, y la proporción ancho/alto.

    La webcam trabaja en espejo: si la imagen del iPhone no lo estaba, se voltea (x -> 1 - x)
    y la etiqueta de MediaPipe se invierte, como si MediaPipe hubiera visto la imagen volteada.
    """
    aspect = vision.image_width / vision.image_height

    def convert(points: list) -> np.ndarray:
        array = np.asarray(points, dtype=float)
        if not vision.mirrored:
            array[:, 0] = 1 - array[:, 0]
        array[:, [0, 2]] *= aspect
        return array

    hands = []
    for hand in vision.hands:
        label = hand.handedness.label
        if not vision.mirrored:
            label = {"Left": "Right", "Right": "Left"}.get(label, label)
        hands.append((convert(hand.landmarks), label == "Right", hand.handedness.score))
    body = convert(vision.pose_landmarks) if vision.pose_landmarks is not None else None
    return hands, body, aspect


def app_text(message: str) -> str:
    for plain, accented in ACCENTS.items():
        message = message.replace(plain, accented)
    return message


@dataclass
class Result:
    """Lo que pasó al terminar una ejecución; se reporta hasta `until_ms`."""

    state: str
    code: str
    message: str
    predicted: str | None
    confidence: float | None
    correct: bool | None
    components: dict[str, str]
    until_ms: float


class PracticeSession:
    def __init__(self, recognizer: Recognizer, catalog: Catalog, target: Sign | None):
        """`target=None`: modo demo (reconoce cualquier seña del catálogo, RF-14)."""
        self.recognizer = recognizer
        self.catalog = catalog
        self.target = target
        self.slots = HandSlots(promote_after_ms=PRIMARY_LOST_MS)
        self.last_t_ms: float | None = None
        self.attempts = 0
        self.correct_attempts = 0
        self.consecutive = 0
        self.approved = False
        self.result: Result | None = None
        self.processing_ms: list[float] = []

    @property
    def practice(self) -> bool:
        return self.target is not None

    def observe(self, observation: ObservationIn) -> dict:
        started = time.perf_counter()
        t_ms = observation.timestamp_ms
        if self.last_t_ms is not None and t_ms <= self.last_t_ms:
            raise ObservationError(
                f"timestamp_ms debe aumentar en cada observación ({t_ms:g} <= {self.last_t_ms:g})"
            )
        self.last_t_ms = t_ms

        hands, body, aspect = to_tracker_units(observation.vision)
        primary, other = self.slots.assign(hands, t_ms)
        detection = primary[:2] if primary else None
        new_sign = self.recognizer.update(
            detection,
            primary[2] if primary else 0.0,
            t_ms,
            other[:2] if other else None,
            body,
        )
        status = body_status(detection[0] if detection else None, body, aspect)

        if new_sign is not None:
            self.result = self._judge(new_sign, status, t_ms)
        elif self.recognizer.rejected:
            self.result = self._needs_both_hands(self.recognizer.rejected, t_ms)
        feedback = self._feedback(observation, detection is not None, status)

        elapsed = (time.perf_counter() - started) * 1000
        self.processing_ms.append(elapsed)
        feedback["processing_time_ms"] = round(elapsed)
        return feedback

    def summary(self, session_id: str) -> dict:
        processing = self.processing_ms or [0.0]
        return {
            "type": "session_summary",
            "session_id": session_id,
            "target_sign": self.target.id if self.target else None,
            "attempts": self.attempts,
            "correct_attempts": self.correct_attempts,
            "approved": self.approved,
            "average_processing_time_ms": round(sum(processing) / len(processing)),
        }

    # --- Resultados de una ejecución --------------------------------------------------------

    def _judge(self, label: str, status: BodyStatus, t_ms: float) -> Result:
        """Se acaba de confirmar `label` (clase del modelo)."""
        sign = self.catalog.find(label)
        predicted = sign.id if sign else None
        confidence = self._confidence(label)
        until = t_ms + RESULT_SHOW_MS
        if not self.practice:
            return Result(
                "confirmed",
                "correct",
                f"Reconocida: {self._name(label)}",
                predicted,
                confidence,
                None,
                self._components("correct", sign),
                until,
            )

        self.attempts += 1
        if label == self.target.data_label and status.framed:
            self.correct_attempts += 1
            self.consecutive += 1
            self.approved = self.consecutive >= APPROVE_AFTER
            return Result(
                "approved" if self.approved else "confirmed",
                "approved" if self.approved else "correct",
                "¡Seña aprobada!"
                if self.approved
                else f"¡Bien! {self.consecutive} de {APPROVE_AFTER}",
                predicted,
                confidence,
                True,
                self._components("correct"),
                until,
            )

        self.consecutive = 0
        if label == self.target.data_label:
            return Result(
                "rejected",
                "adjust_framing",
                f"Seña correcta, pero {app_text(status.issue).lower()}",
                predicted,
                confidence,
                False,
                self._components("correct"),
                until,
            )
        dynamic = self.target.type == "dynamic"
        failed = "movement" if dynamic else "configuration"
        return Result(
            "rejected",
            "wrong_movement" if dynamic else "wrong_configuration",
            f"Se reconoció {self._name(label)}. "
            + ("Revisa el movimiento" if dynamic else "Revisa la forma de tu mano"),
            predicted,
            confidence,
            False,
            self._components("insufficient_data", failed=failed),
            until,
        )

    def _needs_both_hands(self, label: str, t_ms: float) -> Result:
        sign = self.catalog.find(label)
        return Result(
            "rejected",
            "use_both_hands",
            f"{self._name(label)} se hace con las dos manos",
            sign.id if sign else None,
            None,
            False if self.practice else None,
            self._components("insufficient_data", sign),
            t_ms + RESULT_SHOW_MS,
        )

    # --- Mensaje por cuadro -----------------------------------------------------------------

    def _feedback(self, observation: ObservationIn, has_hand: bool, status: BodyStatus) -> dict:
        t_ms = observation.timestamp_ms
        static = self.target is not None and self.target.type == "static"
        base = {
            "type": "feedback",
            "sequence": observation.sequence,
            "timestamp_ms": observation.timestamp_ms,
            "target_sign": self.target.id if self.target else None,
            "correct": None,
            "approved": self.approved if self.practice else None,
            "consecutive_correct": self.consecutive if self.practice else None,
        }

        if self.approved:
            return base | {
                "state": "approved",
                "predicted_sign": self.target.id,
                "confidence": None,
                "progress": 1.0 if static else None,
                "correct": True,
                "components": self._components("correct"),
                "feedback_code": "approved",
                "message": "¡Seña aprobada!",
            }

        result = self.result
        if result is not None and t_ms < result.until_ms:
            return base | {
                "state": result.state,
                "predicted_sign": result.predicted,
                "confidence": result.confidence,
                "progress": (1.0 if result.correct else 0.0) if static else None,
                "correct": result.correct,
                "components": result.components,
                "feedback_code": result.code,
                "message": result.message,
            }

        stabilizer = self.recognizer.stabilizer
        candidate = stabilizer.candidate
        progress = self.recognizer.progress(t_ms)
        frame_confidence = self.recognizer.frame_label[1] if self.recognizer.frame_label else None
        waiting = {
            "state": "waiting",
            "predicted_sign": None,
            "confidence": frame_confidence,
            "progress": 0.0 if static or not self.practice else None,
            "components": self._components("insufficient_data"),
            "feedback_code": None,
        }

        if not has_hand:
            return (
                base
                | waiting
                | {
                    "state": "no_hand",
                    "confidence": None,
                    "feedback_code": "show_hand",
                    "message": "Muestra tu mano a la cámara",
                }
            )
        if status.issue:
            return (
                base
                | waiting
                | {
                    "feedback_code": "adjust_framing",
                    "message": app_text(status.issue),
                }
            )
        if self.recognizer.moving:
            return (
                base
                | waiting
                | {
                    "state": "candidate",
                    "progress": 0.0 if static else None,
                    "message": "Siguiendo el movimiento…",
                }
            )
        if candidate is not None and candidate == stabilizer.confirmed:
            # El estabilizador no vuelve a confirmar la misma seña mientras se sostenga.
            return base | waiting | {"message": "Baja la mano y vuelve a hacer la seña"}

        if not self.practice:
            if candidate is None:
                return base | waiting | {"message": "Haz una seña del catálogo"}
            sign = self.catalog.find(candidate)
            return (
                base
                | waiting
                | {
                    "state": "candidate",
                    "predicted_sign": sign.id if sign else None,
                    "progress": progress,
                    "components": self._components("correct", sign),
                    "feedback_code": "hold_position",
                    "message": "Mantén la posición",
                }
            )

        if not static:
            return (
                base
                | waiting
                | {"message": f"Haz la seña {self.target.display_name} con su movimiento"}
            )
        if candidate == self.target.data_label:
            return (
                base
                | waiting
                | {
                    "state": "candidate",
                    "predicted_sign": self.target.id,
                    "progress": progress,
                    "correct": True,
                    "components": self._components("correct"),
                    "feedback_code": "hold_position",
                    "message": "Mantén la posición",
                }
            )
        if candidate is not None:
            sign = self.catalog.find(candidate)
            return (
                base
                | waiting
                | {
                    "state": "candidate",
                    "predicted_sign": sign.id if sign else None,
                    "correct": False,
                    "components": self._components("insufficient_data", failed="configuration"),
                    "feedback_code": "wrong_configuration",
                    "message": "Revisa la forma de tu mano",
                }
            )
        return base | waiting | {"message": f"Haz la seña {self.target.display_name}"}

    # --- Ayudas -----------------------------------------------------------------------------

    def _components(
        self, status: str, sign: Sign | None = None, failed: str | None = None
    ) -> dict[str, str]:
        """Estado de cada componente de `sign` (por defecto la seña objetivo).

        El modelo evalúa la seña completa: por ahora todos los componentes requeridos
        comparten el resultado, salvo `failed`. La localización queda `not_available` hasta
        definir las zonas con la persona intérprete.
        """
        sign = sign or self.target
        required = sign.required_components if sign else ()
        components = {}
        for name in COMPONENTS:
            if name not in required:
                components[name] = "not_required"
            elif name == failed:
                components[name] = "incorrect"
            elif name == "localization" and status == "correct":
                components[name] = "not_available"
            else:
                components[name] = status
        return components

    def _confidence(self, label: str) -> float | None:
        movement = self.recognizer.last_movement
        if movement and movement[0] == label:
            return round(movement[1], 3)
        frame = self.recognizer.frame_label
        return round(frame[1], 3) if frame and frame[0] == label else None

    def _name(self, label: str) -> str:
        sign = self.catalog.find(label)
        return sign.display_name if sign else label
