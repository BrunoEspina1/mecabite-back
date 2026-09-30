import numpy as np
import pytest

from app.vision import body as b

ASPECT = 16 / 9


def frontal_body(shift_y: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Persona de frente y centrada en una imagen 16:9 (unidades de la altura)."""
    center_x = ASPECT / 2
    points = np.zeros((33, 4))
    points[:, 3] = 1.0

    def put(index: int, dx: float, y: float) -> None:
        points[index, :2] = (center_x + dx * scale, 0.6 + (y - 0.6) * scale + shift_y)

    put(b.NOSE, 0.0, 0.35)
    put(b.LEFT_EYE, 0.04, 0.31)
    put(b.RIGHT_EYE, -0.04, 0.31)
    put(b.LEFT_EAR, 0.1, 0.33)
    put(b.RIGHT_EAR, -0.1, 0.33)
    put(b.MOUTH_LEFT, 0.03, 0.42)
    put(b.MOUTH_RIGHT, -0.03, 0.42)
    put(b.LEFT_SHOULDER, 0.25, 0.6)
    put(b.RIGHT_SHOULDER, -0.25, 0.6)
    return points


def hand_at(x: float, y: float) -> np.ndarray:
    return np.tile([x, y, 0.0], (21, 1))


def test_frontal_body_is_framed() -> None:
    assert b.framing_issue(frontal_body(), ASPECT) is None


@pytest.mark.parametrize(
    ("index", "column", "value", "expected"),
    [
        (b.NOSE, 3, 0.1, "No se ve tu cara"),
        (b.LEFT_SHOULDER, 3, 0.1, "no se ven tus dos hombros"),
        (b.LEFT_SHOULDER, 0, -0.1, "no se ven tus dos hombros"),
        (b.NOSE, 0, ASPECT / 2 + 0.2, "Ponte de frente"),
    ],
)
def test_framing_explains_what_to_fix(index: int, column: int, value: float, expected: str) -> None:
    body = frontal_body()
    body[index, column] = value
    assert expected in b.framing_issue(body, ASPECT)


def test_framing_without_body() -> None:
    assert "No se ve tu cuerpo" in b.framing_issue(None, ASPECT)


def test_framing_asks_to_step_back_or_closer() -> None:
    assert "no se ve tu pecho" in b.framing_issue(frontal_body(shift_y=0.3), ASPECT)
    assert "Acercate" in b.framing_issue(frontal_body(scale=0.3), ASPECT)


@pytest.mark.parametrize(
    ("x", "y", "zone"),
    [
        (0.0, 0.25, "frente"),
        (0.0, 0.35, "cara"),
        (0.0, 0.43, "boca/barbilla"),
        (0.2, 0.35, "lado de la cara"),
        (0.25, 0.6, "hombro"),
        (0.0, 0.8, "pecho"),
        (-0.7, 0.9, "lejos del cuerpo"),
    ],
)
def test_hand_location_by_body_zone(x: float, y: float, zone: str) -> None:
    hand = hand_at(ASPECT / 2 + x, y)
    assert b.hand_location(hand, frontal_body()) == zone


def test_hand_location_needs_body() -> None:
    assert b.hand_location(hand_at(0.9, 0.5), None) is None
    status = b.body_status(None, None, ASPECT)
    assert not status.framed


def with_wrists(body: np.ndarray, right_y: float, left_y: float) -> np.ndarray:
    """Cada muñeca bajo su hombro, a la altura indicada."""
    body = body.copy()
    body[b.RIGHT_WRIST, :2] = (body[b.RIGHT_SHOULDER, 0], right_y)
    body[b.LEFT_WRIST, :2] = (body[b.LEFT_SHOULDER, 0], left_y)
    return body


def test_other_hand_counts_only_when_raised() -> None:
    body = frontal_body()  # hombros en y=0.6, 0.5 de ancho
    hand = hand_at(ASPECT / 2 - 0.25, 0.7)  # la mano que se mueve, cerca de la muñeca derecha
    raised, hanging = with_wrists(body, 0.7, left_y=0.9), with_wrists(body, 0.7, left_y=1.4)

    # Aunque el detector de manos solo vea una (manos juntas), el pose ve la otra muñeca.
    assert b.other_hand_raised(hand, None, raised)
    assert not b.other_hand_raised(hand, None, hanging)
    # La otra mano a la vista pero colgando no cuenta.
    assert not b.other_hand_raised(hand, hand_at(ASPECT / 2 + 0.25, 1.4), hanging)
    assert b.other_hand_raised(hand, hand_at(ASPECT / 2 + 0.25, 0.9), hanging)


def test_other_hand_without_a_reliable_body_is_just_being_seen() -> None:
    hand, other = hand_at(0.5, 0.5), hand_at(0.9, 0.5)
    tiny = frontal_body(scale=0.2)  # video de solo la mano: hombros inventados y pequeños

    assert b.other_hand_raised(hand, other, None)
    assert not b.other_hand_raised(hand, None, None)
    assert b.other_hand_raised(hand, other, tiny)
    assert not b.other_hand_raised(None, other, None)
