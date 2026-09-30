import numpy as np
import pytest

from app.vision.catalog import OTHER
from app.vision.recognizer import (
    HOLD_SECONDS,
    VOTE_MS,
    VOTE_SHARE,
    Recognizer,
    SignStabilizer,
    alphabet_key,
)
from app.vision.sequence import MAX_MOVEMENT_DURATION_MS, DynamicDetector

HAND = (np.zeros((21, 3)), False)


class FakeClassifier:
    """Siempre predice la misma clase: aísla la lógica de votos y tiempos del modelo."""

    def __init__(self, label: str, confidence: float = 0.9):
        self.labels = [label, OTHER]
        self.prediction = (label, confidence)

    def predict(self, _vector: np.ndarray) -> tuple[str, float]:
        return self.prediction


def first_confirmation(
    recognizer: Recognizer, fps: float, seconds: float = 2.0, detection=HAND
) -> float | None:
    """Alimenta cuadros a `fps` y devuelve el t_ms en que se confirma una seña."""
    for index in range(int(seconds * fps)):
        t_ms = index * 1000 / fps
        if recognizer.update(detection, 1.0, t_ms) is not None:
            return t_ms
    return None


@pytest.mark.parametrize("fps", [30, 15])
def test_static_sign_confirms_at_the_same_time_at_any_frame_rate(fps: int) -> None:
    confirmed_at = first_confirmation(Recognizer(FakeClassifier("A")), fps)

    # Casi una ventana de votos y luego sostenerla; a lo más dos cuadros de diferencia.
    expected = VOTE_MS * VOTE_SHARE + HOLD_SECONDS * 1000
    assert confirmed_at is not None
    assert expected <= confirmed_at <= expected + 2 * 1000 / fps


def test_sign_confirms_once_while_held() -> None:
    recognizer = Recognizer(FakeClassifier("A"))

    confirmations = [recognizer.update(HAND, 1.0, t_ms) for t_ms in range(0, 3000, 33)]

    assert [sign for sign in confirmations if sign] == ["A"]
    assert recognizer.progress(3000) == 0.0


@pytest.mark.parametrize("classifier", [FakeClassifier("A", confidence=0.3), FakeClassifier(OTHER)])
def test_unsure_or_other_predictions_never_confirm(classifier: FakeClassifier) -> None:
    assert first_confirmation(Recognizer(classifier), fps=30) is None


def test_without_hand_nothing_is_recognized() -> None:
    recognizer = Recognizer(FakeClassifier("A"))

    assert first_confirmation(recognizer, fps=30, detection=None) is None
    assert recognizer.frame_label is None
    assert recognizer.is_left is None


def test_hand_side_needs_a_confident_handedness() -> None:
    recognizer = Recognizer(FakeClassifier("A"))
    left_hand = (np.zeros((21, 3)), True)

    recognizer.update(left_hand, 0.5, 0)
    assert recognizer.is_left is None
    recognizer.update(left_hand, 0.95, 33)
    assert recognizer.is_left is True


def test_signs_lists_model_classes_without_other() -> None:
    assert Recognizer(FakeClassifier("Ñ")).signs == ["Ñ"]


def test_dynamic_sign_is_classified_after_two_and_a_half_seconds_if_still_moving() -> None:
    detector = DynamicDetector(FakeClassifier("hola"), OTHER)
    started_at = None
    confirmed_at = None

    for t_ms in range(0, 3001, 50):
        points = hand(0.4 + t_ms / 1000 * 0.5)
        result = detector.push(t_ms, points)
        if detector.moving and started_at is None:
            started_at = t_ms
        if result is not None:
            confirmed_at = t_ms
            break

    assert started_at is not None
    assert confirmed_at is not None
    assert confirmed_at - started_at >= MAX_MOVEMENT_DURATION_MS
    assert confirmed_at - started_at < MAX_MOVEMENT_DURATION_MS + 50


def test_load_without_models_explains_how_to_train(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="vision train"):
        Recognizer.load(tmp_path / "static.joblib", tmp_path / "dynamic.joblib")


def test_stabilizer_confirms_after_holding_one_second() -> None:
    stabilizer = SignStabilizer(hold_seconds=1.0)

    assert stabilizer.update("I", 0.0) is None
    assert stabilizer.update("J", 0.4) is None  # cambio a mitad del movimiento: reinicia
    assert stabilizer.update("J", 1.3) is None
    assert stabilizer.update("J", 1.5) == "J"
    assert stabilizer.update("J", 3.0) is None  # mantenerla no la repite
    assert stabilizer.confirmed == "J"


def test_stabilizer_clears_after_one_second_without_sign() -> None:
    stabilizer = SignStabilizer(hold_seconds=1.0)
    stabilizer.update("A", 0.0)
    stabilizer.update("A", 1.0)

    stabilizer.update(None, 1.5)
    assert stabilizer.confirmed == "A"
    stabilizer.update(None, 2.5)
    assert stabilizer.confirmed is None


def test_stabilizer_ignores_a_brief_slip() -> None:
    stabilizer = SignStabilizer(hold_seconds=1.0, grace_seconds=0.3)
    stabilizer.update("A", 0.0)
    stabilizer.update("A", 0.3)  # también una seña nueva debe durar la tolerancia

    assert stabilizer.update("B", 0.5) is None  # la mano se movió un poco
    assert stabilizer.update("B", 0.7) is None
    assert stabilizer.candidate == "A"
    assert stabilizer.update("A", 0.8) is None
    assert stabilizer.update("A", 1.0) == "A"  # el segundo cuenta desde 0, no desde 0.8


def test_stabilizer_counts_a_real_change_from_when_it_started() -> None:
    stabilizer = SignStabilizer(hold_seconds=1.0, grace_seconds=0.3)
    stabilizer.update("A", 0.0)

    stabilizer.update("B", 0.5)
    assert stabilizer.update("B", 0.9) is None
    assert (stabilizer.candidate, stabilizer.since) == ("B", 0.5)
    assert stabilizer.update("B", 1.5) == "B"


def test_alphabet_order_puts_enie_after_n() -> None:
    assert sorted(["O", "Ñ", "A", "N"], key=alphabet_key) == ["A", "N", "Ñ", "O"]


# --- Señas de dos manos ----------------------------------------------------------------------


def hand(x: float, y: float = 0.5) -> np.ndarray:
    """Mano con la palma de 0.1 (muñeca -> nudillo medio) con la muñeca en (x, y)."""
    points = np.zeros((21, 3))
    points[:, :2] = (x, y)
    points[9, 1] -= 0.1
    return points


def sliding(t_ms: float) -> np.ndarray:
    """Quieta, se desliza 0.6 (seis palmas) entre 500 y 1000 ms y vuelve a quedarse quieta."""
    return hand(0.4 + 0.6 * np.clip((t_ms - 500) / 500, 0, 1))


def still(_t_ms: float) -> np.ndarray:
    return hand(1.3)


def two_hands_recognizer(moving_label: str = "gracias") -> Recognizer:
    dynamic = FakeClassifier(moving_label)
    two_handed = frozenset({"gracias"})
    return Recognizer(
        FakeClassifier(OTHER),
        DynamicDetector(dynamic, OTHER, two_handed),
        DynamicDetector(dynamic, OTHER, two_handed, two_handed_only=True),
    )


def play(recognizer: Recognizer, primary, second=None) -> tuple[list[str], list[str]]:
    """Dos segundos a 30 fps. Devuelve las señas confirmadas y las descartadas por una mano."""
    confirmed, rejected = [], []
    for t_ms in np.arange(0, 2000, 1000 / 30):
        other = (second(t_ms), True) if second else None
        sign = recognizer.update((primary(t_ms), False), 1.0, t_ms, other)
        confirmed += [sign] if sign else []
        rejected += [recognizer.rejected] if recognizer.rejected else []
    return confirmed, rejected


def test_two_handed_sign_with_one_hand_is_rejected() -> None:
    assert play(two_hands_recognizer(), sliding) == ([], ["gracias"])


def test_two_handed_sign_counts_once_with_the_other_hand() -> None:
    assert play(two_hands_recognizer(), sliding, still) == (["gracias"], [])


def test_two_handed_sign_made_by_the_second_hand() -> None:
    # Gracias: la mano que apareció primero se queda quieta de base y la otra se mueve.
    recognizer = two_hands_recognizer()

    assert play(recognizer, still, sliding) == (["gracias"], [])
    assert len(recognizer.moving_detector.frames()) > 0
    assert recognizer.moving_detector is recognizer.second_detector


def test_letters_are_only_read_from_the_main_hand() -> None:
    assert play(two_hands_recognizer("J"), still, sliding) == ([], [])
    assert play(two_hands_recognizer("J"), sliding, still) == (["J"], [])
