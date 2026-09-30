"""Inclinación del guante con cada seña bien hecha: la referencia de las correcciones.

El MPU no marca "palma al frente" en grados absolutos (depende de cómo quedó montado), así que
se compara contra lo que midió el guante con la seña correcta. Es la del guante de la mano
derecha (GLOVE_RIGHT_EMITTER): la del izquierdo quedaría en espejo y no se usa. Sale de las
grabaciones de `vision practica_guante` (data/practice/glove/<persona>/<seña>/<toma>/glove.csv)
o de sostener la seña en vivo. Se guarda en app/catalog/glove_reference.json (separado del
código, RNF-11).

También compara los dedos grabados con `expected.glove_fingers` del catálogo para ajustarlo.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

from app.catalog import Catalog, get_catalog
from app.core.config import settings
from app.sessions.feedback import REFERENCE_PATH
from app.vision.glove import FINGERS, GLOVE_VALUES, GloveFeed, GloveReading

GLOVE_DIR = Path(__file__).resolve().parents[2] / "data" / "practice" / "glove"


def read_recording(path: Path, emitter: int) -> list[GloveReading]:
    with path.open(newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))
    return [
        GloveReading.from_values(float(row["t_ms"]), [float(row[name]) for name in GLOVE_VALUES])
        for row in rows
        if int(float(row["emisor"])) == emitter
    ]


def summarize(readings: list[GloveReading]) -> dict:
    return {
        "roll": round(median(r.roll for r in readings), 1),
        "pitch": round(median(r.pitch for r in readings), 1),
        "fingers": {
            finger: Counter(round(r.fingers[finger]) for r in readings).most_common(1)[0][0]
            for finger in FINGERS
        },
        "samples": len(readings),
    }


def from_recordings(
    catalog: Catalog, glove_dir: Path = GLOVE_DIR, emitter: int = 1
) -> dict[str, dict]:
    """Referencia por id del catálogo con todas las grabaciones de cada seña."""
    by_sign: dict[str, list[GloveReading]] = {}
    for path in sorted(glove_dir.glob("*/*/*/glove.csv")):
        sign = catalog.find(path.parent.parent.name)
        if sign is None:
            continue
        by_sign.setdefault(sign.id, []).extend(read_recording(path, emitter))
    return {sign_id: summarize(readings) for sign_id, readings in by_sign.items() if readings}


def from_live(sign_id: str, seconds: float, emitter: int = 1) -> dict:
    feed = GloveFeed(settings.glove_device_name, settings.glove_characteristic_uuid)
    feed.start()
    try:
        while not feed.recent(500, emitter):
            if feed.error:
                print(f"Error BLE: {feed.error}")
            time.sleep(0.5)
        input(f"Guante conectado. Haz la seña {sign_id!r} y presiona Enter para medir...")
        time.sleep(seconds)
        readings = feed.recent(seconds * 1000, emitter)
    finally:
        feed.close()
    if not readings:
        raise SystemExit("No llegaron lecturas del guante")
    return summarize(readings)


def report(catalog: Catalog, reference: dict[str, dict]) -> None:
    """Muestra la referencia y los dedos grabados que el catálogo no aceptaría."""
    for sign in catalog.signs:
        data = reference.get(sign.id)
        if data is None:
            continue
        print(
            f"{sign.display_name:10} inclinación={data['pitch']:6.1f}  lado={data['roll']:6.1f}  "
            f"dedos={data['fingers']}  ({data['samples']} lecturas)"
        )
        for finger, allowed in sign.expected.glove_fingers.items():
            if data["fingers"][finger] not in allowed:
                print(
                    f"   ! {finger}: grabado {data['fingers'][finger]}, el catálogo acepta "
                    f"{list(allowed)} (revisa expected.glove_fingers en signs.json)"
                )


def save(reference: dict[str, dict], path: Path = REFERENCE_PATH) -> None:
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8")).get("signs", {})
    data = {
        "_nota": (
            "Inclinación (grados) del guante con cada seña bien hecha. roll = der - izq, "
            "pitch = arr - abj. La generan `vision glove-reference`; las correcciones avisan si "
            "la mano se aleja más de GLOVE_TILT_TOLERANCE_DEG."
        ),
        "updated": datetime.now().isoformat(timespec="seconds"),
        "signs": existing | reference,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Guardado en {path}")


def run(sign: str | None, live: bool, seconds: float, glove_dir: Path) -> None:
    catalog = get_catalog()
    emitter = settings.glove_right_emitter
    if live:
        found = catalog.find(sign or "")
        if found is None:
            raise SystemExit("Con --live indica la seña: --sign a")
        reference = {found.id: from_live(found.id, seconds, emitter)}
    else:
        reference = from_recordings(catalog, glove_dir, emitter)
        if sign:
            found = catalog.find(sign)
            reference = {k: v for k, v in reference.items() if found and k == found.id}
        if not reference:
            raise SystemExit(f"No hay grabaciones del guante de señas del catálogo en {glove_dir}")
    report(catalog, reference)
    save(reference)
