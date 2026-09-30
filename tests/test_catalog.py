import json

import pytest
from fastapi.testclient import TestClient

from app.catalog import CATALOG_PATH, get_catalog, load_catalog
from app.core.config import settings
from app.vision.catalog import OTHER
from app.vision.catalog import load_catalog as load_vision_catalog


def test_catalog_has_five_signs_per_level() -> None:
    catalog = get_catalog()

    assert [sign.id for sign in catalog.signs if sign.level == 1] == ["a", "b", "c", "l", "y"]
    assert [sign.id for sign in catalog.signs if sign.level == 2] == ["j", "enie", "q", "x", "z"]
    assert [sign.id for sign in catalog.signs if sign.level == 3] == [
        "hola",
        "gracias",
        "por_favor",
        "ayuda",
        "mama",
    ]


def test_level_defines_components_and_type_defines_timing() -> None:
    catalog = get_catalog()
    a, j, hola = catalog.get("a"), catalog.get("j"), catalog.get("hola")

    assert a.required_components == ("configuration", "orientation")
    # El tiempo de las estáticas sale de .env (VISION_HOLD_MS), el mismo que usa el reconocedor.
    assert (a.hold_time_ms, a.max_duration_ms) == (settings.vision_hold_ms, None)
    assert "movement" in j.required_components
    assert (j.hold_time_ms, j.max_duration_ms) == (None, 3000)
    assert "localization" in hola.required_components
    assert (a.hands, hola.hands, catalog.get("gracias").hands) == (1, 1, 2)
    assert set(a.components) == {"configuration", "orientation", "localization", "movement"}


def test_find_accepts_id_display_name_and_data_folder() -> None:
    catalog = get_catalog()

    assert catalog.find("enie").id == "enie"
    assert catalog.find("Ñ").id == "enie"
    assert catalog.find("Ñ").id == "enie"  # Ñ descompuesta, como la guarda macOS
    assert catalog.find("Por favor").id == "por_favor"
    assert catalog.find("D") is None


def test_catalog_rejects_duplicated_ids(tmp_path) -> None:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    data["signs"].append(data["signs"][0])
    path = tmp_path / "signs.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="repetidos"):
        load_catalog(path)


@pytest.mark.parametrize(
    ("expected", "error"),
    [
        ({"glove_margin": 1.5}, "glove_margin"),
        ({"tilt_tolerance_deg": 0}, "tilt_tolerance_deg"),
        ({"glove_confirms": True}, "glove_confirms necesita glove_fingers"),
    ],
)
def test_catalog_rejects_bad_glove_settings(tmp_path, expected: dict, error: str) -> None:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    data["signs"][0]["expected"] = expected
    path = tmp_path / "signs.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_catalog(path)


def test_vision_catalog_maps_folders_to_training_classes() -> None:
    catalog = load_vision_catalog(extras_path=None)

    assert catalog.static_class("A") == "A"
    assert catalog.static_class("D") == OTHER
    assert catalog.dynamic_class("Ñ") == "Ñ"
    assert catalog.dynamic_class("K") == OTHER
    assert "K" in catalog.dynamic_folders
    assert catalog.two_handed == {"gracias", "por_favor"}


def test_vision_catalog_adds_training_only_alphabet() -> None:
    catalog = load_vision_catalog()

    assert catalog.static_class("A") == "A"
    assert catalog.static_class("D") == "D"
    assert catalog.static_class("reposo") == OTHER
    assert catalog.dynamic_class("K") == "K"
    assert "K" not in catalog.dynamic_negative
    assert load_catalog().find("D") is None  # la API no la anuncia


def test_catalog_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/catalog/signs")

    assert response.status_code == 200
    body = response.json()
    assert body["catalog_version"] == get_catalog().version
    assert [level["level"] for level in body["levels"]] == [1, 2, 3]
    assert len(body["signs"]) == 15
    enie = next(sign for sign in body["signs"] if sign["id"] == "enie")
    assert enie["display_name"] == "Ñ"
    assert enie["type"] == "dynamic"
    assert enie["max_duration_ms"] == 3000
    assert enie["reference_asset"] == "enie.mp4"
    assert {sign["id"] for sign in body["signs"] if sign["hands"] == 2} == {"gracias", "por_favor"}
