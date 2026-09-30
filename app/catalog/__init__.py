"""Catálogo de señas (RF-09), separado del código en signs.json (RNF-11).

Es la única fuente de ids, niveles y componentes de cada seña: lo usan la API y, a través
de `app.vision.catalog`, el entrenamiento de visión.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from app.core.config import settings

CATALOG_PATH = Path(__file__).with_name("signs.json")
COMPONENTS = ("configuration", "orientation", "localization", "movement")
SIGN_ID = re.compile(r"^[a-z0-9_]+$")
FINGERS = ("pulgar", "indice", "medio", "anular", "menique")
FINGER_STATES = ("extended", "half", "flexed")
PALMS = ("facing", "side")
POINTING = ("up", "down")


def _key(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().casefold()


@dataclass(frozen=True)
class Level:
    level: int
    name: str
    required_components: tuple[str, ...]


@dataclass(frozen=True)
class Expected:
    """Cómo debe verse la seña para dar correcciones concretas (`expected` en signs.json)."""

    # Valores del guante aceptados por dedo (1 encogido, 2 a medias, 3 estirado).
    glove_fingers: dict[str, tuple[int, ...]]
    # Estado de cada dedo revisado con la cámara: extended | half | flexed.
    camera_fingers: dict[str, str]
    palm: str | None  # facing | side
    pointing: str | None  # up | down
    # Cuánto puede pasarse un dedo de `glove_fingers` antes de pedir corregirlo.
    glove_margin: float = 0.25
    # Grados que la inclinación puede alejarse de la referencia (None: GLOVE_TILT_TOLERANCE_DEG).
    tilt_tolerance_deg: float | None = None
    # En práctica, el guante solo (dedos e inclinación sostenidos) cuenta la seña aunque la
    # cámara no la confirme o no vea la mano.
    glove_confirms: bool = False


NO_EXPECTATION = Expected({}, {}, None, None)


@dataclass(frozen=True)
class Sign:
    id: str
    display_name: str
    level: int
    type: str
    data_label: str
    required_components: tuple[str, ...]
    hold_time_ms: int | None
    max_duration_ms: int | None
    reference_asset: str | None
    hands: int  # 2 si la seña usa las dos manos
    # Descripción de los cuatro componentes; None si no aplica o falta validarla.
    components: dict[str, str | None]
    validated: bool
    expected: Expected = NO_EXPECTATION


@dataclass(frozen=True)
class Catalog:
    version: str
    levels: tuple[Level, ...]
    signs: tuple[Sign, ...]
    # Señas con movimiento fuera del catálogo (ej. K): negativos del detector dinámico.
    dynamic_negatives: frozenset[str]

    def get(self, sign_id: str) -> Sign | None:
        return next((sign for sign in self.signs if sign.id == sign_id), None)

    def find(self, text: str) -> Sign | None:
        """Busca por id, nombre mostrado o carpeta de datos: `enie`, `Ñ` y `ñ` dan la Ñ.

        Sirve para traducir la clase de un modelo (`Ñ`) al id del protocolo (`enie`).
        """
        key = _key(text)
        for sign in self.signs:
            if key in {_key(sign.id), _key(sign.display_name), _key(sign.data_label)}:
                return sign
        return None


def _expected(sign_id: str, item: dict) -> Expected:
    def fail(message: str) -> ValueError:
        return ValueError(f"`expected` de {sign_id!r}: {message}")

    unknown = set(item) - {
        "glove_fingers",
        "camera_fingers",
        "palm",
        "pointing",
        "glove_margin",
        "tilt_tolerance_deg",
        "glove_confirms",
    }
    if unknown:
        raise fail(f"campos desconocidos {unknown}")
    glove = item.get("glove_fingers", {})
    camera = item.get("camera_fingers", {})
    for finger in set(glove) | set(camera):
        if finger not in FINGERS:
            raise fail(f"dedo desconocido {finger!r}")
    for finger, values in glove.items():
        if not values or not set(values) <= {1, 2, 3}:
            raise fail(f"{finger} debe aceptar valores entre 1 y 3, no {values!r}")
    for finger, state in camera.items():
        if state not in FINGER_STATES:
            raise fail(f"{finger} debe ser {' | '.join(FINGER_STATES)}, no {state!r}")
    if item.get("palm") not in (None, *PALMS):
        raise fail(f"palm debe ser {' | '.join(PALMS)}")
    if item.get("pointing") not in (None, *POINTING):
        raise fail(f"pointing debe ser {' | '.join(POINTING)}")
    margin = item.get("glove_margin", 0.25)
    if not isinstance(margin, (int, float)) or not 0 <= margin < 1:
        raise fail(f"glove_margin debe estar entre 0 y 1, no {margin!r}")
    tolerance = item.get("tilt_tolerance_deg")
    if tolerance is not None and (not isinstance(tolerance, (int, float)) or tolerance <= 0):
        raise fail(f"tilt_tolerance_deg debe ser positivo, no {tolerance!r}")
    confirms = item.get("glove_confirms", False)
    if not isinstance(confirms, bool):
        raise fail("glove_confirms debe ser true o false")
    if confirms and not glove:
        raise fail("glove_confirms necesita glove_fingers")
    return Expected(
        glove_fingers={finger: tuple(sorted(values)) for finger, values in glove.items()},
        camera_fingers=dict(camera),
        palm=item.get("palm"),
        pointing=item.get("pointing"),
        glove_margin=float(margin),
        tilt_tolerance_deg=None if tolerance is None else float(tolerance),
        glove_confirms=confirms,
    )


def load_catalog(path: Path = CATALOG_PATH) -> Catalog:
    data = json.loads(path.read_text(encoding="utf-8"))
    levels = {
        item["level"]: Level(item["level"], item["name"], tuple(item["required_components"]))
        for item in data["levels"]
    }
    sign_types = data["sign_types"]

    signs = []
    for item in data["signs"]:
        sign_id = item["id"]
        if not SIGN_ID.match(sign_id):
            raise ValueError(f"Id de seña inválido {sign_id!r}: usa minúsculas ASCII y _")
        if item["level"] not in levels:
            raise ValueError(f"La seña {sign_id!r} usa el nivel {item['level']}, que no existe")
        if item["type"] not in sign_types:
            raise ValueError(f"La seña {sign_id!r} tiene un tipo desconocido: {item['type']!r}")
        if item.get("hands", 1) not in (1, 2):
            raise ValueError(f"La seña {sign_id!r} debe usar 1 o 2 manos, no {item['hands']!r}")
        unknown = set(item.get("components", {})) - set(COMPONENTS)
        if unknown:
            raise ValueError(f"La seña {sign_id!r} tiene componentes desconocidos: {unknown}")

        timing = sign_types[item["type"]]
        signs.append(
            Sign(
                id=sign_id,
                display_name=unicodedata.normalize("NFC", item["display_name"]),
                level=item["level"],
                type=item["type"],
                data_label=unicodedata.normalize("NFC", item.get("data_label", sign_id)),
                required_components=levels[item["level"]].required_components,
                # Se configura en .env (VISION_HOLD_MS): la app lo lee de aquí para su barra de
                # progreso, así siempre coincide con el tiempo que usa el reconocedor.
                hold_time_ms=settings.vision_hold_ms if item["type"] == "static" else None,
                max_duration_ms=timing["max_duration_ms"],
                reference_asset=item.get("reference_asset"),
                hands=item.get("hands", 1),
                components={name: item.get("components", {}).get(name) for name in COMPONENTS},
                validated=item.get("validated", False),
                expected=_expected(sign_id, item.get("expected", {})),
            )
        )

    ids = [sign.id for sign in signs]
    duplicated = {sign_id for sign_id in ids if ids.count(sign_id) > 1}
    if duplicated:
        raise ValueError(f"Ids de seña repetidos: {sorted(duplicated)}")

    return Catalog(
        version=data["catalog_version"],
        levels=tuple(levels[number] for number in sorted(levels)),
        signs=tuple(signs),
        dynamic_negatives=frozenset(
            unicodedata.normalize("NFC", label)
            for label in data.get("training", {}).get("dynamic_negatives", [])
        ),
    )


@cache
def get_catalog() -> Catalog:
    return load_catalog()
