"""Correcciones concretas con guante y cámara (`app/sessions/feedback.py`)."""

import json

import numpy as np
import pytest

from app.catalog import CATALOG_PATH, get_catalog, load_catalog
from app.sessions.feedback import (
    CorrectionFilter,
    GloveState,
    camera_finger_states,
    evaluate,
)
from app.vision.glove import GLOVE_VALUES, GloveReading

CATALOG = get_catalog()
WRIST = np.array([0.5, 0.8, 0.0])
MCP_X = {"indice": -0.04, "medio": -0.013, "anular": 0.013, "menique": 0.04}
MCP_INDEX = {"indice": 5, "medio": 9, "anular": 13, "menique": 17}


def hand(
    bends: dict[str, float] | None = None,
    thumb: str = "extended",
    side: bool = False,
    down: bool = False,
) -> np.ndarray:
    """Mano derecha sintética con los dedos hacia arriba; `bends`: radianes por dedo."""
    bends = {"indice": 0.0, "medio": 0.0, "anular": 0.0, "menique": 0.0} | (bends or {})
    points = np.zeros((21, 3))
    points[0] = WRIST
    for finger, mcp in MCP_INDEX.items():
        current = WRIST + [MCP_X[finger], -0.1, 0.0]
        points[mcp] = current
        angle = 0.0
        for joint in (1, 2, 3):
            angle += bends[finger] / 3
            current = current + 0.03 * np.array([0.0, -np.cos(angle), np.sin(angle)])
            points[mcp + joint] = current
    tip = {"extended": (-0.12, -0.08), "flexed": (0.0, -0.06)}[thumb]
    for joint in (1, 2, 3, 4):
        points[joint] = WRIST + [tip[0] * joint / 4, tip[1] * joint / 4, 0.0]
    if side:
        points[:, 0] = WRIST[0] + (points[:, 0] - WRIST[0]) * 0.2
    if down:
        points[:, 1] = WRIST[1] - (points[:, 1] - WRIST[1])
    return points


def glove(pitch: float = 0.0, roll: float = 0.0, **fingers: float) -> GloveState:
    named = {"pulgar": 3, "indice": 3, "medio": 3, "anular": 3, "menique": 3} | fingers
    return GloveState(fingers=named, roll=roll, pitch=pitch)


def messages(sign_id: str, **kwargs) -> list[str]:
    sign = CATALOG.get(sign_id)
    args = {"hand": None, "body": None, "glove": None} | kwargs
    return [c.message for c in evaluate(sign, **args)]


A_FIST = {"pulgar": 3, "indice": 1, "medio": 2, "anular": 1, "menique": 1}


def test_glove_says_which_finger_to_stretch_or_bend() -> None:
    assert messages("b", glove=glove(pulgar=1, anular=2)) == ["Estira más el dedo anular"]
    assert messages("a", glove=glove(**A_FIST | {"menique": 3})) == ["Encoge más el meñique"]


def test_a_well_made_a_has_no_corrections() -> None:
    fist = hand({"indice": 4.0, "medio": 4.0, "anular": 4.0, "menique": 4.0})
    assert messages("a", glove=glove(**A_FIST), hand=fist) == []


def test_middle_finger_reading_two_counts_as_bent() -> None:
    """El sensor del medio no llega a 1: con 2 ya está encogido."""
    assert messages("a", glove=glove(**A_FIST | {"medio": 2})) == []
    assert messages("a", glove=glove(**A_FIST | {"medio": 3})) == ["Encoge más el dedo medio"]


def test_three_wrong_fingers_become_one_instruction_for_the_hand() -> None:
    assert messages("a", glove=glove(pulgar=3)) == ["Cierra más la mano: encoge los dedos"]


def test_c_ignores_the_glove_and_uses_the_camera() -> None:
    """En la C el guante marca todos los dedos en 3: se revisa 100 % con la cámara."""
    curved = hand({"indice": 2.0, "medio": 2.0, "anular": 2.0, "menique": 2.0}, side=True)
    assert messages("c", glove=glove(), hand=curved) == []

    straight = hand(side=True)
    assert messages("c", glove=glove(), hand=straight) == ["Curva más los dedos"]


def test_camera_finger_states() -> None:
    states = camera_finger_states(hand({"indice": 4.0, "medio": 2.0}, thumb="flexed"))
    assert states == {
        "indice": "flexed",
        "medio": "half",
        "anular": "extended",
        "menique": "extended",
        "pulgar": "flexed",
    }


def test_l_needs_the_thumb_out_seen_by_the_camera() -> None:
    """El pulgar del guante marca 2 muy seguido: la L revisa el pulgar con la cámara."""
    l_glove = glove(pulgar=2, indice=3, medio=2, anular=1, menique=1)
    bent = {"medio": 4.0, "anular": 4.0, "menique": 4.0}
    assert messages("l", glove=l_glove, hand=hand(bent)) == []
    assert messages("l", glove=l_glove, hand=hand(bent, thumb="flexed")) == ["Estira más el pulgar"]


def test_camera_orientation() -> None:
    fist = {"indice": 4.0, "medio": 4.0, "anular": 4.0, "menique": 4.0}
    assert messages("a", hand=hand(fist, side=True)) == ["Gira la palma hacia la cámara"]
    assert messages("a", hand=hand(fist, down=True)) == ["Apunta los dedos hacia arriba"]
    curved = {"indice": 2.0, "medio": 2.0, "anular": 2.0, "menique": 2.0}
    assert messages("c", hand=hand(curved)) == [
        "Gira la mano de lado: la palma mira hacia un costado"
    ]


def test_glove_tilt_against_the_reference() -> None:
    reference = {"pitch": -43.0, "roll": 0.0}
    fist = glove(**A_FIST)
    assert messages("a", glove=fist, reference=reference) == ["Inclina la mano hacia abajo"]

    tilted = glove(pitch=-40.0, roll=-45.0, **A_FIST)
    assert messages("a", glove=tilted, reference=reference) == ["Gira la muñeca hacia la derecha"]


def test_low_hand_is_only_a_hint_to_raise_the_arm() -> None:
    body = np.zeros((33, 4))
    body[:, 3] = 1.0
    body[11, :2] = (0.7, 0.4)
    body[12, :2] = (0.3, 0.4)
    low = hand({"indice": 4.0, "medio": 4.0, "anular": 4.0, "menique": 4.0}) + [0, 0.2, 0]
    corrections = evaluate(CATALOG.get("a"), low, body, None)
    assert [(c.part, c.hint) for c in corrections] == [("arm", True)]


def test_filter_waits_before_showing_and_tolerates_a_blink() -> None:
    [correction] = evaluate(CATALOG.get("a"), None, None, glove(**A_FIST | {"menique": 3}))
    corrections_filter = CorrectionFilter(hold_ms=400, grace_ms=250)

    assert corrections_filter.update([correction], 0) == []
    assert corrections_filter.update([correction], 300) == []
    assert corrections_filter.update([correction], 450) == [correction]
    assert corrections_filter.update([], 600) == [correction], "un parpadeo no la quita"
    assert corrections_filter.update([], 800) == []


def test_glove_state_uses_the_median_of_recent_readings() -> None:
    values = dict.fromkeys(GLOVE_VALUES, 0.0) | {"abj": 60.0, "der": 10.0, "indice": 3.0}
    noisy = values | {"indice": 1.0, "abj": 0.0}
    readings = [
        GloveReading.from_values(t, [v[name] for name in GLOVE_VALUES])
        for t, v in enumerate([values, values, noisy])
    ]
    state = GloveState.from_readings(readings)
    assert (state.pitch, state.roll, state.fingers["indice"]) == (-60.0, 10.0, 3.0)


def test_catalog_rejects_invalid_expectations(tmp_path) -> None:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    data["signs"][0]["expected"] = {"glove_fingers": {"indice": [4]}}
    path = tmp_path / "signs.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="indice"):
        load_catalog(path)


@pytest.mark.parametrize(
    ("finger", "name"),
    [
        ("pulgar", "el pulgar"),
        ("indice", "el dedo índice"),
        ("medio", "el dedo medio"),
        ("anular", "el dedo anular"),
        ("menique", "el meñique"),
    ],
)
def test_every_finger_gets_its_own_correction(finger: str, name: str) -> None:
    a_ok = glove(**A_FIST)
    stretched = glove(**A_FIST | {finger: 3})
    bent = glove(**A_FIST | {finger: 1})
    wrong = {"pulgar": bent}.get(finger, stretched)
    expected = f"Estira más {name}" if finger == "pulgar" else f"Encoge más {name}"

    assert messages("a", glove=a_ok) == []
    assert messages("a", glove=wrong) == [expected]
    assert messages("b", glove=glove(pulgar=1, **{finger: 1} if finger != "pulgar" else {})) == (
        [f"Estira más {name}"] if finger != "pulgar" else []
    )


def test_every_catalog_sign_but_mama_checks_all_five_fingers() -> None:
    for sign in CATALOG.signs:
        checked = set(sign.expected.glove_fingers) | set(sign.expected.camera_fingers)
        missing = {"pulgar", "indice", "medio", "anular", "menique"} - checked
        # C: el guante marca todo en 3 y el pulgar por cámara es poco confiable. Mamá: falta
        # confirmar su configuración con la persona intérprete.
        assert missing == {"c": {"pulgar"}, "mama": missing}.get(sign.id, set()), sign.id
