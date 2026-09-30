"""Registro en consola de lo que está entrando: las dos manos y los dos guantes.

Lo usan `vision demo/live/practice/practica_guante --log` y la API con `INPUT_LOG=true`. Imprime
una línea cada `every_ms` (no una por cuadro) para poder leerla en vivo:

    [12:03:04.512] manos: principal=derecha (A 0.93) · otra=izquierda | guante der E1: dedos
    3-1-2-1-1 roll -4 pitch -43 | guante izq E2: sin datos
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Protocol

from app.vision.glove import FINGERS

SIDES = {False: "derecha", True: "izquierda"}


class GloveLike(Protocol):
    """Una lectura del guante (GloveReading) o la mediana de varias (GloveState)."""

    fingers: dict[str, float]
    roll: float
    pitch: float


def describe_hand(hand: tuple | None, label: tuple[str, float] | None = None) -> str:
    """`hand`: (puntos, izquierda, ...) como los entrega HandTracker; `label`: (seña, confianza)."""
    if hand is None:
        return "no se ve"
    text = SIDES[bool(hand[1])]
    if label is not None:
        text += f" ({label[0]} {label[1]:.2f})"
    return text


def describe_glove(name: str, emitter: int, reading: GloveLike | None) -> str:
    if reading is None:
        return f"guante {name} E{emitter}: sin datos"
    fingers = "-".join(f"{reading.fingers[finger]:g}" for finger in FINGERS)
    return (
        f"guante {name} E{emitter}: dedos {fingers} "
        f"roll {reading.roll:.0f} pitch {reading.pitch:.0f}"
    )


class InputLog:
    """Imprime el estado de las entradas como mucho una vez cada `every_ms`."""

    def __init__(self, every_ms: float = 500, prefix: str = "", out=print):
        self.every_ms = every_ms
        self.prefix = prefix
        self.out = out
        self._last_ms: float | None = None

    def due(self, now_ms: float | None = None) -> bool:
        now_ms = time.time() * 1000 if now_ms is None else now_ms
        if self._last_ms is not None and now_ms - self._last_ms < self.every_ms:
            return False
        self._last_ms = now_ms
        return True

    def write(
        self,
        primary: tuple | None,
        other: tuple | None,
        gloves: list[tuple[str, int, GloveLike | None]],
        label: tuple[str, float] | None = None,
        extra: str = "",
        now_ms: float | None = None,
    ) -> None:
        """`gloves`: (nombre, emisor, última lectura) de cada guante; vacío si no se usan."""
        if not self.due(now_ms):
            return
        clock = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        parts = [f"manos: principal={describe_hand(primary, label)} · otra={describe_hand(other)}"]
        parts += [describe_glove(*glove) for glove in gloves]
        if extra:
            parts.append(extra)
        self.out(f"[{clock}] {self.prefix}" + " | ".join(parts), flush=True)
