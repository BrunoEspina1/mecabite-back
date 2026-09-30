"""Registro de las entradas (`--log` en la CLI, INPUT_LOG en la API)."""

import numpy as np

from app.vision.glove import GloveReading
from app.vision.input_log import InputLog


def test_one_line_with_both_hands_and_both_gloves() -> None:
    lines = []
    log = InputLog(out=lambda text, flush: lines.append(text))
    right = GloveReading.from_values(0, [4, 0, 0, 43, 0, 44, 1, 3, 2, 1, 1])
    hand = (np.zeros((21, 3)), False)

    log.write(hand, None, [("der", 1, right), ("izq", 2, None)], ("A", 0.93), now_ms=0)

    assert lines[0].endswith(
        "manos: principal=derecha (A 0.93) · otra=no se ve"
        " | guante der E1: dedos 1-3-2-1-1 roll -4 pitch -43 | guante izq E2: sin datos"
    )


def test_it_writes_at_most_twice_a_second() -> None:
    lines = []
    log = InputLog(every_ms=500, out=lambda text, flush: lines.append(text))
    for now_ms in range(0, 1200, 33):
        log.write(None, None, [], now_ms=now_ms)
    assert len(lines) == 3  # 0, ~500 y ~1000 ms


def test_the_collector_gives_the_latest_reading_of_each_glove() -> None:
    import time

    from app.vision.glove import GloveCollector

    collector = GloveCollector("GuanteLSM", "uuid")
    now = int(time.time() * 1000)
    collector._store((now - 2000, 2, [0] * 6 + [1, 1, 1, 1, 1]))  # viejo: ya no cuenta
    collector._store((now - 100, 1, [0] * 6 + [3, 3, 3, 3, 3]))
    collector._store((now - 50, 1, [0] * 6 + [1, 3, 3, 3, 3]))

    assert collector.latest(1).fingers["pulgar"] == 1
    assert collector.latest(2) is None
