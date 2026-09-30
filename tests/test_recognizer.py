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


def test_alphabet_order_puts_enie_after_n() -> None:
    assert sorted(["O", "Ñ", "A", "N"], key=alphabet_key) == ["A", "N", "Ñ", "O"]
