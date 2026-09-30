"""POST /sessions + WebSocket (contrato móvil 0.2.0), con un clasificador falso: sin modelos."""

from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.v1.routes.sessions import get_model_loader
from app.main import app
from app.sessions.practice import WRONG_SIGN_MS, to_tracker_units
from app.sessions.protocol import VisionIn
from app.sessions.store import ModelNotAvailable, Models, SessionStore, get_store
from app.vision.catalog import OTHER
from app.vision.recognizer import HOLD_SECONDS, VOTE_MS

WIDTH, HEIGHT = 720, 1280
FRAME_MS = 33


class FakeClassifier:
    """Siempre predice la misma clase: aísla la sesión del modelo real."""

    def __init__(self, label: str = "A", confidence: float = 0.9):
        self.labels = [label, OTHER]
        self.prediction = (label, confidence)

    def predict(self, _vector: np.ndarray) -> tuple[str, float]:
        return self.prediction


def framed_body() -> list[list[float]]:
    """De frente, centrado, con cara y hombros a la vista (coordenadas normalizadas)."""
    body = [[0.5, 0.55, 0.0, 1.0] for _ in range(33)]
    body[0] = [0.5, 0.3, 0.0, 1.0]  # nariz
    body[11] = [0.3, 0.5, 0.0, 1.0]  # hombros
    body[12] = [0.7, 0.5, 0.0, 1.0]
    return body


HAND = {
    "landmarks": [[0.5 + i * 0.001, 0.6 - i * 0.002, 0.0] for i in range(21)],
    "handedness": {"label": "Right", "score": 0.95},
}


def observation(sequence: int, t_ms: float, hand: bool = True, body: bool = True) -> dict:
    return {
        "type": "observation",
        "sequence": sequence,
        "timestamp_ms": t_ms,
        "vision": {
            "image_width": WIDTH,
            "image_height": HEIGHT,
            "mirrored": True,
            "hands": [HAND] if hand else [],
            "pose_landmarks": framed_body() if body else None,
        },
        "glove": None,
    }


def session_request(target: str | None = "a", mode: str = "practice", **extra) -> dict:
    return {
        "mode": mode,
        "target_sign": target,
        "participant_id": "p01",
        "device_id": "test",
        "client_version": "0.2.0",
        "calibration_id": None,
        "record": False,
    } | extra


@pytest.fixture
def models() -> Models:
    return Models(static=FakeClassifier(), dynamic=None, version="vision-test")


@pytest.fixture
def client(models: Models) -> Iterator[TestClient]:
    store = SessionStore()

    async def load() -> Models:
        if models is None:
            raise ModelNotAvailable("No hay modelo de visión en el servidor")
        return models

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_model_loader] = lambda: load
    yield TestClient(app)
    app.dependency_overrides.clear()


class Player:
    """Manda cuadros a 30 fps con tiempo creciente y guarda cada feedback."""

    def __init__(self, websocket):
        self.websocket = websocket
        self.sequence = 0
        self.t_ms = 0.0

    def frames(self, seconds: float, **kwargs) -> list[dict]:
        replies = []
        for _ in range(int(seconds * 1000 / FRAME_MS)):
            self.websocket.send_json(observation(self.sequence, self.t_ms, **kwargs))
            replies.append(self.websocket.receive_json())
            self.sequence += 1
            self.t_ms += FRAME_MS
        return replies


def create(client: TestClient, **kwargs) -> dict:
    response = client.post("/api/v1/sessions", json=session_request(**kwargs))
    assert response.status_code == 201, response.json()
    return response.json()


def test_create_session_returns_the_websocket_path(client: TestClient) -> None:
    session = create(client)

    assert session["session_id"].startswith("sess_")
    assert session["websocket_path"] == f"/api/v1/ws/sessions/{session['session_id']}"
    assert (session["mode"], session["target_sign"], session["level"]) == ("practice", "a", 1)
    assert (session["protocol_version"], session["model_version"]) == ("0.2.0", "vision-test")


def test_three_framed_executions_in_a_row_approve_the_sign(client: TestClient) -> None:
    session = create(client)
    with client.websocket_connect(session["websocket_path"]) as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["required_inputs"] == ["hand", "pose"]
        player = Player(websocket)

        codes = []
        for _ in range(3):
            replies = player.frames(VOTE_MS / 1000 + HOLD_SECONDS + 0.2)
            codes.append([reply["feedback_code"] for reply in replies])
            assert "hold_position" in codes[-1]
            assert all(reply["progress"] is not None for reply in replies)
            player.frames(HOLD_SECONDS + 0.3, hand=False)  # bajar la mano para repetir

        assert "correct" in codes[0] and "correct" in codes[1]
        assert codes[2][-1] == "approved"
        last = player.frames(0.1)[-1]
        assert (last["state"], last["approved"], last["consecutive_correct"]) == (
            "approved",
            True,
            3,
        )

        websocket.send_json({"type": "end_session", "reason": "user_finished"})
        summary = websocket.receive_json()
        assert summary == summary | {
            "type": "session_summary",
            "target_sign": "a",
            "attempts": 3,
            "correct_attempts": 3,
            "approved": True,
        }

    # Al cerrarla, la sesión deja de existir.
    with (
        client.websocket_connect(session["websocket_path"]) as websocket,
        pytest.raises(WebSocketDisconnect) as closed,
    ):
        websocket.receive_json()
    assert closed.value.code == 4404


def test_a_failed_attempt_keeps_the_correct_count(client: TestClient, models: Models) -> None:
    session = create(client)
    attempt = VOTE_MS / 1000 + HOLD_SECONDS + 0.2
    with client.websocket_connect(session["websocket_path"]) as websocket:
        websocket.receive_json()
        player = Player(websocket)
        player.frames(attempt)
        player.frames(HOLD_SECONDS + 0.3, hand=False)
        models.static.prediction = ("B", 0.9)
        failed = player.frames(attempt + 0.5)
        player.frames(HOLD_SECONDS + 0.3, hand=False)
        models.static.prediction = ("A", 0.9)
        second = player.frames(attempt)

    rejected = next(reply for reply in failed if reply["state"] == "rejected")
    assert rejected["message"].startswith("Se reconoció B")
    assert rejected["consecutive_correct"] == 1
    assert any(reply["message"] == "¡Bien! 2 de 3" for reply in second)


def test_holding_after_a_confirmation_asks_to_repeat(client: TestClient) -> None:
    session = create(client)
    with client.websocket_connect(session["websocket_path"]) as websocket:
        websocket.receive_json()
        replies = Player(websocket).frames(VOTE_MS / 1000 + HOLD_SECONDS + 2.0)

    first = next(i for i, reply in enumerate(replies) if reply["state"] == "confirmed")
    assert replies[first]["message"] == "¡Bien! 1 de 3"
    assert replies[-1]["message"] == "Baja la mano y vuelve a hacer la seña"
    assert replies[-1]["consecutive_correct"] == 1


def test_a_correct_sign_without_the_body_in_frame_does_not_count(client: TestClient) -> None:
    session = create(client)
    with client.websocket_connect(session["websocket_path"]) as websocket:
        websocket.receive_json()
        replies = Player(websocket).frames(VOTE_MS / 1000 + HOLD_SECONDS + 0.2, body=False)

    assert replies[0]["feedback_code"] == "adjust_framing"
    rejected = next(reply for reply in replies if reply["state"] == "rejected")
    assert rejected["feedback_code"] == "adjust_framing"
    assert rejected["message"].startswith("Seña correcta, pero no se ve tu cuerpo")
    assert rejected["consecutive_correct"] == 0


def test_moving_the_hand_a_little_does_not_fail_the_attempt(
    client: TestClient, models: Models
) -> None:
    session = create(client)
    with client.websocket_connect(session["websocket_path"]) as websocket:
        websocket.receive_json()
        player = Player(websocket)
        player.frames(0.6)
        switched_at = player.t_ms
        models.static.prediction = ("B", 0.9)  # otra forma de mano
        replies = player.frames(1.2)

    wrong = [reply for reply in replies if reply["feedback_code"] == "wrong_configuration"]
    assert wrong, "si la otra forma se sostiene, sí se avisa"
    assert wrong[0]["timestamp_ms"] - switched_at >= WRONG_SIGN_MS
    assert all(reply["state"] != "rejected" for reply in replies)
    assert replies[-1]["consecutive_correct"] == 0


WITH_MOVEMENT = Models(FakeClassifier("A"), FakeClassifier("J"), "vision-test")


@pytest.mark.parametrize("models", [WITH_MOVEMENT])
def test_static_targets_do_not_look_for_movement(client: TestClient) -> None:
    store = app.dependency_overrides[get_store]()

    static = store.get(create(client, target="a")["session_id"])
    dynamic = store.get(create(client, target="j")["session_id"])

    assert static.practice.recognizer.detector is None
    assert dynamic.practice.recognizer.detector is not None


@pytest.mark.parametrize("models", [WITH_MOVEMENT])
def test_held_shapes_do_not_count_against_a_moving_sign(client: TestClient) -> None:
    session = create(client, target="j")
    with client.websocket_connect(session["websocket_path"]) as websocket:
        websocket.receive_json()
        replies = Player(websocket).frames(VOTE_MS / 1000 + HOLD_SECONDS + 1.0)
        websocket.send_json({"type": "end_session", "reason": "user_finished"})
        summary = websocket.receive_json()

    assert all(reply["state"] != "rejected" for reply in replies)
    assert replies[-1]["message"] == "Haz la seña J con su movimiento"
    assert summary["attempts"] == 0


def test_no_hand_asks_to_show_it(client: TestClient) -> None:
    session = create(client)
    with client.websocket_connect(session["websocket_path"]) as websocket:
        websocket.receive_json()
        reply = Player(websocket).frames(0.05, hand=False)[0]

    assert (reply["state"], reply["feedback_code"]) == ("no_hand", "show_hand")
    assert reply["components"] == {
        "configuration": "insufficient_data",
        "orientation": "insufficient_data",
        "localization": "not_required",
        "movement": "not_required",
    }


def test_invalid_observations_answer_an_error_and_keep_the_socket_open(client: TestClient) -> None:
    session = create(client)
    with client.websocket_connect(session["websocket_path"]) as websocket:
        websocket.receive_json()
        bad = observation(0, 0)
        bad["vision"]["hands"][0] = HAND | {"landmarks": HAND["landmarks"][:20]}
        websocket.send_json(bad)
        error = websocket.receive_json()
        assert (error["type"], error["code"], error["sequence"]) == (
            "error",
            "INVALID_OBSERVATION",
            0,
        )

        websocket.send_json(observation(1, 100))
        assert websocket.receive_json()["type"] == "feedback"
        websocket.send_json(observation(2, 100))  # el tiempo no avanzó
        assert websocket.receive_json()["code"] == "INVALID_OBSERVATION"
        websocket.send_text("no es json")
        assert websocket.receive_json()["code"] == "INVALID_REQUEST"
        websocket.send_json(observation(3, 133))
        assert websocket.receive_json()["sequence"] == 3


def test_unknown_session_closes_with_4404(client: TestClient) -> None:
    with (
        client.websocket_connect("/api/v1/ws/sessions/sess_nope") as websocket,
        pytest.raises(WebSocketDisconnect) as closed,
    ):
        websocket.receive_json()
    assert closed.value.code == 4404


def test_reconnecting_keeps_the_same_session(client: TestClient) -> None:
    session = create(client)
    with client.websocket_connect(session["websocket_path"]) as websocket:
        websocket.receive_json()
        Player(websocket).frames(0.1)
    with client.websocket_connect(session["websocket_path"]) as websocket:
        assert websocket.receive_json()["session_id"] == session["session_id"]
        websocket.send_json(observation(10, 1000))
        assert websocket.receive_json()["sequence"] == 10


def test_demo_reports_the_recognized_sign(client: TestClient) -> None:
    session = create(client, target=None, mode="demo")
    assert (session["target_sign"], session["level"]) == (None, None)
    with client.websocket_connect(session["websocket_path"]) as websocket:
        websocket.receive_json()
        replies = Player(websocket).frames(VOTE_MS / 1000 + HOLD_SECONDS + 0.2)

    confirmed = next(reply for reply in replies if reply["state"] == "confirmed")
    assert (confirmed["predicted_sign"], confirmed["message"]) == ("a", "Reconocida: A")
    assert (confirmed["target_sign"], confirmed["correct"], confirmed["approved"]) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("request_body", "status", "code"),
    [
        (session_request("zz"), 404, "SIGN_NOT_FOUND"),
        (session_request(None), 400, "INVALID_REQUEST"),
        (session_request(client_version="1.0.0"), 426, "PROTOCOL_VERSION_UNSUPPORTED"),
        ({"mode": "practice", "target_sign": "a"}, 400, "INVALID_REQUEST"),
        # Sin modelo dinámico el servidor no reconoce la J.
        (session_request("j"), 503, "SIGN_NOT_TRAINED"),
    ],
)
def test_create_session_errors(client: TestClient, request_body: dict, status: int, code: str):
    response = client.post("/api/v1/sessions", json=request_body)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["message"]


@pytest.mark.parametrize("models", [None])
def test_create_session_without_a_trained_model(client: TestClient) -> None:
    response = client.post("/api/v1/sessions", json=session_request())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_AVAILABLE"


def test_unmirrored_frames_are_flipped_like_the_webcam() -> None:
    vision = observation(0, 0)["vision"] | {"mirrored": False}
    hands, body, aspect = to_tracker_units(VisionIn.model_validate(vision))

    points, is_left, _score = hands[0]
    assert aspect == WIDTH / HEIGHT
    # x -> 1 - x y luego en alturas de imagen; MediaPipe habría dicho "Left" en espejo.
    assert points[0, 0] == pytest.approx((1 - 0.5) * aspect)
    assert points[0, 1] == pytest.approx(0.6)
    assert is_left is False
    assert body is not None and body.shape == (33, 4)
