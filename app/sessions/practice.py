"""Una sesión de práctica o demostración: observaciones de la app -> mensajes `feedback`.

Las reglas son las de `vision live --target` (`app/vision/teacher.py`), ahora sin cámara:
una ejecución de la seña objetivo cuenta solo si la persona está bien encuadrada (de frente,
con cara y hombros a la vista) y 3 correctas aprueban la seña. A diferencia de `live` (y de
RF-10, "consecutivas"), un intento fallido no reinicia la cuenta: se decidió así para no
castigar un error después de dos aciertos.

Solo se comparan señas del mismo tipo que la objetivo: al practicar una estática no se buscan
movimientos (acomodar la mano no es una J) y al practicar una con movimiento no cuentan las
formas quietas del camino (la I al inicio de la J).

Los puntos llegan normalizados a la imagen vertical que vio MediaPipe en el iPhone. Se
convierten a las unidades de `HandTracker`/`BodyTracker` (vista en espejo, x y z en alturas
de imagen), las mismas con las que se entrenaron los modelos.

El guante (por BLE en la laptop, o en la observación) y la cámara se comparan con `expected` de
la seña objetivo para dar correcciones concretas (`corrections`: "Estira más el dedo anular").
Una ejecución que el modelo reconoce como la objetivo no cuenta si hay correcciones de un
componente requerido: se rechaza diciendo qué corregir (RF-13, principio Indivisa).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from app.catalog import COMPONENTS, Catalog, Sign
from app.core.config import settings
from app.sessions.feedback import (
    Correction,
    CorrectionFilter,
    GloveState,
    evaluate,
    glove_reference,
)
from app.sessions.protocol import GloveIn, ObservationIn, VisionIn
from app.vision.body import BodyStatus, body_status
from app.vision.glove import FINGERS, GLOVE_VALUES, GloveReading
from app.vision.recognizer import Recognizer
from app.vision.tracker import LABEL_MIN_SCORE, PRIMARY_LOST_MS, HandSlots

APPROVE_AFTER = 3  # correctas para aprobar; las fallidas no reinician la cuenta
RESULT_SHOW_MS = 1500  # tiempo que se sigue reportando un resultado para que la app lo muestre
WRONG_SIGN_MS = settings.vision_wrong_sign_ms  # otra seña sostenida esto antes de avisar
# body.py escribe sin acentos (las fuentes de OpenCV solo tienen ASCII); la app sí los tiene.
ACCENTS = {"camara": "cámara", "Alejate": "Aléjate", "Acercate": "Acércate", "Muevete": "Muévete"}
MAX_CORRECTIONS = 2  # correcciones que se mandan a la vez: más confunden
MOVEMENT_WINDOW_MS = 2000  # lecturas del guante que cuentan para una seña con movimiento
WRONG_CODE = {
    "configuration": "wrong_configuration",
    "orientation": "wrong_orientation",
    "localization": "wrong_localization",
}
# Pide las lecturas de un guante (emisor) de los últimos N ms (reloj del servidor).
GloveSource = Callable[[float, int], list[GloveReading]]


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


def glove_readings(glove: GloveIn) -> list[GloveReading]:
    """La lectura que trae la observación (pruebas y datos simulados)."""
    if not glove.connected:
        return []
    values = glove.values
    if isinstance(values, dict):
        missing = [finger for finger in FINGERS if finger not in values]
        if missing:
            raise ObservationError(f"glove.values: faltan los dedos {', '.join(missing)}")
        values = [float(values.get(name, 0.0)) for name in GLOVE_VALUES]
    return [GloveReading.from_values(time.time() * 1000, values)]


def glove_json(state: GloveState | None) -> dict:
    return {"connected": state is not None} | (state.to_json() if state else {})


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
    corrections: list[Correction] = field(default_factory=list)


class PracticeSession:
    def __init__(
        self,
        recognizer: Recognizer,
        catalog: Catalog,
        target: Sign | None,
        glove_source: GloveSource | None = None,
        glove_required: bool | None = None,
        dominant_hand: str | None = None,
    ):
        """`target=None`: modo demo (reconoce cualquier seña del catálogo, RF-14).

        `glove_source`: lecturas recientes del guante por BLE; la lectura de la observación, si
        viene, tiene prioridad. Con `glove_required` la práctica no evalúa sin guante.

        `dominant_hand` ("left"/"right"): la mano de referencia. Con las dos manos a la vista es
        la principal (la que se reconoce y se corrige); con una sola, se usa la que se ve. Su
        lado se toma de la elección de la persona y no de la etiqueta de MediaPipe, que se
        equivoca con el puño o de perfil y haría comparar la mano en espejo con los modelos.
        """
        self.recognizer = recognizer
        self.catalog = catalog
        self.target = target
        self.glove_source = glove_source
        self.glove_required = settings.glove_required if glove_required is None else glove_required
        self.dominant_left = None if dominant_hand is None else dominant_hand == "left"
        # Un guante por mano: se corrige con el de la mano dominante (derecha si no se eligió).
        self.glove_emitter, self.other_glove_emitter = (
            (settings.glove_left_emitter, settings.glove_right_emitter)
            if self.dominant_left
            else (settings.glove_right_emitter, settings.glove_left_emitter)
        )
        self.glove: GloveState | None = None
        self.other_glove: GloveState | None = None
        self.glove_history: deque[tuple[float, GloveState]] = deque(maxlen=512)
        # La inclinación de referencia se grabó con el guante derecho: con el izquierdo no aplica.
        self.reference = (
            glove_reference().get(target.id) if target and not self.dominant_left else None
        )
        self.filter = CorrectionFilter()
        self.live: list[Correction] = []  # correcciones estables del momento
        self.slots = HandSlots(promote_after_ms=PRIMARY_LOST_MS, dominant_left=self.dominant_left)
        # La única mano a la vista es la no dominante: se reconoce, pero no se corrige (el guante
        # y `expected` describen la dominante).
        self.only_other_hand = False
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
        primary, other = self.slots.assign(hands, t_ms, locked=self.recognizer.moving)
        primary, other = self._with_sides(primary, other)
        detection = primary[:2] if primary else None
        new_sign = self.recognizer.update(
            detection,
            primary[2] if primary else 0.0,
            t_ms,
            other[:2] if other else None,
            body,
        )
        status = body_status(detection[0] if detection else None, body, aspect)
        self.glove = self._read_glove(observation, t_ms)
        self.live = self._live_corrections(detection, body, t_ms)

        if new_sign is not None:
            self.result = self._judge(new_sign, status, t_ms) or self.result
        elif self.recognizer.rejected:
            self.result = self._needs_both_hands(self.recognizer.rejected, t_ms)
        feedback = self._feedback(observation, detection is not None, status)
        feedback["corrections"] = [
            correction.to_json() for correction in self._shown(t_ms, detection is not None, status)
        ]
        feedback["glove"] = glove_json(self.glove)
        feedback["other_glove"] = glove_json(self.other_glove)

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

    def _judge(self, label: str, status: BodyStatus, t_ms: float) -> Result | None:
        """Se acaba de confirmar `label` (clase del modelo). None si no cuenta como intento."""
        if self.practice and self.recognizer.is_dynamic(label) != (self.target.type == "dynamic"):
            return None
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

        if self.glove_required and self.glove is None:
            self._hold_again(t_ms, 0)  # sin guante no se evalúa: el feedback dice `disconnected`
            return None
        self.attempts += 1
        if label == self.target.data_label and status.framed:
            blocking = self._blocking(self._attempt_corrections(t_ms))
            if blocking:
                self._hold_again(t_ms, RESULT_SHOW_MS)
                return Result(
                    "rejected",
                    WRONG_CODE[blocking[0].component],
                    blocking[0].message,
                    predicted,
                    confidence,
                    False,
                    self._components("correct", failed={c.component for c in blocking}),
                    until,
                    blocking,
                )
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
        blocking = self._blocking(self.live)
        if blocking:
            return Result(
                "rejected",
                WRONG_CODE[blocking[0].component],
                f"Se reconoció {self._name(label)}. {blocking[0].message}",
                predicted,
                confidence,
                False,
                self._components("insufficient_data", failed={c.component for c in blocking}),
                until,
                blocking,
            )
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

        if self.practice and self.glove_required and self.glove is None:
            return (
                base
                | waiting
                | {
                    "state": "disconnected",
                    "message": "Conecta el guante: no llegan sus datos",
                }
            )
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
        # El estabilizador no vuelve a confirmar la misma seña mientras se sostenga.
        held_again = candidate is not None and candidate == stabilizer.confirmed
        if held_again and (static or not self.practice):
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

        blocking = self._blocking(self.live)
        if blocking:
            sign = self.catalog.find(candidate) if candidate else None
            return (
                base
                | waiting
                | {
                    "state": "candidate" if candidate else "waiting",
                    "predicted_sign": sign.id if sign else None,
                    "correct": False,
                    "components": self._components(
                        "insufficient_data", failed={c.component for c in blocking}
                    ),
                    "feedback_code": WRONG_CODE[blocking[0].component],
                    "message": blocking[0].message,
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
        # Acomodar la mano cambia la seña un instante: se avisa solo si la otra se sostiene.
        if candidate is not None and (t_ms / 1000 - stabilizer.since) * 1000 >= WRONG_SIGN_MS:
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

    def _read_glove(self, observation: ObservationIn, t_ms: float) -> GloveState | None:
        """Guante de la mano dominante (el de la observación tiene prioridad) y el de la otra."""
        window = settings.glove_max_age_ms
        source = self.glove_source
        self.other_glove = GloveState.from_readings(
            source(window, self.other_glove_emitter) if source else []
        )
        if observation.glove is not None:
            readings = glove_readings(observation.glove)
        elif source is not None:
            readings = source(window, self.glove_emitter)
        else:
            readings = []
        state = GloveState.from_readings(readings)
        if state is not None:
            self.glove_history.append((t_ms, state))
        return state

    def _with_sides(self, primary: TrackedHand | None, other: TrackedHand | None) -> tuple:
        """Pone a cada mano el lado que eligió la persona (ver `dominant_hand`)."""
        self.only_other_hand = False
        if self.dominant_left is None or primary is None:
            return primary, other
        points, is_left, score = primary
        if other is None and is_left != self.dominant_left and score >= LABEL_MIN_SCORE:
            self.only_other_hand = True
            return primary, other
        primary = (points, self.dominant_left, score)
        if other is not None:
            other = (other[0], not self.dominant_left, other[2])
        return primary, other

    def _live_corrections(
        self, detection: tuple[np.ndarray, bool] | None, body: np.ndarray | None, t_ms: float
    ) -> list[Correction]:
        """Correcciones que se mantienen; durante un movimiento solo se revisan los dedos."""
        if not self.practice or self.approved or detection is None or self.only_other_hand:
            return self.filter.update([], t_ms)
        still = not self.recognizer.moving
        corrections = evaluate(
            self.target,
            detection[0] if still else None,
            body if still else None,
            self.glove,
            self.reference if still else None,
        )
        return self.filter.update(corrections, t_ms)

    def _attempt_corrections(self, t_ms: float) -> list[Correction]:
        """Qué falló en la ejecución que se acaba de reconocer.

        Estática: el guante de este momento (ya es la mediana de los últimos ms; un dedo que se
        movió al final todavía no pasa el filtro de parpadeo) más las correcciones estables.
        Con movimiento: el guante a lo largo del trazo, contra la forma e inclinación de la seña.
        """
        if self.only_other_hand:
            return []
        if self.target.type == "static":
            fresh = evaluate(self.target, None, None, self.glove, self.reference)
            seen = {c.key for c in fresh}
            return fresh + [c for c in self.live if c.key not in seen]
        states = [state for when, state in self.glove_history if t_ms - when <= MOVEMENT_WINDOW_MS]
        if not states:
            return []
        movement = GloveState(
            fingers={f: float(np.median([s.fingers[f] for s in states])) for f in FINGERS},
            roll=float(np.median([s.roll for s in states])),
            pitch=float(np.median([s.pitch for s in states])),
        )
        return evaluate(self.target, None, None, movement, self.reference)

    def _blocking(self, corrections: list[Correction]) -> list[Correction]:
        """Correcciones que hacen fallar el intento: de componentes que evalúa el nivel."""
        required = self.target.required_components if self.target else ()
        return [
            c
            for c in corrections
            if not c.hint
            and c.component in required
            and (c.source != "camera" or settings.feedback_camera_blocks)
        ]

    def _hold_again(self, t_ms: float, after_ms: float) -> None:
        """La estática se vuelve a evaluar al sostenerla otra vez, pasados `after_ms`: así se
        corrige sin bajar la mano (tras un rechazo, o al conectarse el guante)."""
        if self.target.type != "static":
            return
        stabilizer = self.recognizer.stabilizer
        stabilizer.confirmed = None
        stabilizer.since = (t_ms + after_ms) / 1000

    def _shown(self, t_ms: float, has_hand: bool, status: BodyStatus) -> list[Correction]:
        if self.approved:
            return []
        if self.result is not None and t_ms < self.result.until_ms:
            return self.result.corrections[:MAX_CORRECTIONS]
        if not has_hand or status.issue:
            return []
        return self.live[:MAX_CORRECTIONS]

    def _components(
        self, status: str, sign: Sign | None = None, failed: str | set[str] | None = None
    ) -> dict[str, str]:
        """Estado de cada componente de `sign` (por defecto la seña objetivo).

        El modelo evalúa la seña completa: por ahora todos los componentes requeridos
        comparten el resultado, salvo `failed`. La localización queda `not_available` hasta
        definir las zonas con la persona intérprete.
        """
        sign = sign or self.target
        required = sign.required_components if sign else ()
        failed = {failed} if isinstance(failed, str) else failed or set()
        components = {}
        for name in COMPONENTS:
            if name not in required:
                components[name] = "not_required"
            elif name in failed:
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
