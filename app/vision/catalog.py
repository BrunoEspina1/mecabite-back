"""Vista del catálogo (app/catalog/signs.json) para entrenar los modelos de visión.

Las clases de los modelos son los nombres de carpeta de data/vision (`A`, `Ñ`);
`app.catalog.Catalog.find` los traduce a los ids del protocolo (`a`, `enie`).

Además del catálogo de la app, los modelos aprenden las señas de training_signs.json (el
resto del abecedario). Esas no están en la API: `find` devuelve None para ellas.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from app.catalog import CATALOG_PATH
from app.catalog import load_catalog as load_signs

OTHER = "otra"
EXTRA_SIGNS_PATH = Path(__file__).with_name("training_signs.json")


@dataclass(frozen=True)
class Catalog:
    static: frozenset[str]
    dynamic: frozenset[str]
    # Señas con movimiento que no están en el catálogo (ej. K): negativos del detector dinámico.
    dynamic_negative: frozenset[str]

    @property
    def dynamic_folders(self) -> frozenset[str]:
        return self.dynamic | self.dynamic_negative

    def static_class(self, folder: str) -> str:
        """Carpetas fuera del catálogo (otras letras, reposo) se aprenden como `otra`."""
        return folder if folder in self.static else OTHER

    def dynamic_class(self, folder: str) -> str:
        return folder if folder in self.dynamic else OTHER


def load_catalog(path: Path = CATALOG_PATH, extras_path: Path | None = EXTRA_SIGNS_PATH) -> Catalog:
    """Con `extras_path=None` quedan solo las señas del catálogo de la app."""
    signs = load_signs(path)
    extras = json.loads(extras_path.read_text(encoding="utf-8")) if extras_path else {}
    extra_static = _nfc(extras.get("estaticas", []))
    extra_dynamic = _nfc(extras.get("dinamicas", []))
    return Catalog(
        static=frozenset(sign.data_label for sign in signs.signs if sign.type == "static")
        | extra_static,
        dynamic=frozenset(sign.data_label for sign in signs.signs if sign.type == "dynamic")
        | extra_dynamic,
        # Una seña extra deja de ser negativo: ahora es una clase (ej. K).
        dynamic_negative=signs.dynamic_negatives - extra_dynamic,
    )


def _nfc(labels: list[str]) -> frozenset[str]:
    return frozenset(unicodedata.normalize("NFC", label) for label in labels)
