"""Sesiones en memoria (el backend no las conserva al cerrarlas) y modelos cargados una vez."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import cache

from fastapi import WebSocket

from app.sessions.practice import PracticeSession
from app.vision.classifier import SignClassifier
from app.vision.recognizer import DYNAMIC_MODEL_PATH, STATIC_MODEL_PATH, Recognizer

SESSION_TTL_SECONDS = 15 * 60  # sin actividad


class ModelNotAvailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Models:
    static: SignClassifier
    dynamic: SignClassifier | None
    version: str

    def new_recognizer(self) -> Recognizer:
        return Recognizer.from_classifiers(self.static, self.dynamic)


@cache
def load_models() -> Models:
    """Se cargan al crear la primera sesión. Tras `vision train` hay que reiniciar la API."""
    if not STATIC_MODEL_PATH.exists():
        raise ModelNotAvailable(
            "No hay modelo de visión en el servidor: ejecuta `vision train` y reinicia la API"
        )
    trained = datetime.fromtimestamp(STATIC_MODEL_PATH.stat().st_mtime)
    return Models(
        static=SignClassifier.load(STATIC_MODEL_PATH),
        dynamic=SignClassifier.load(DYNAMIC_MODEL_PATH) if DYNAMIC_MODEL_PATH.exists() else None,
        version=f"vision-{trained:%Y%m%d.%H%M}",
    )


@dataclass
class StoredSession:
    id: str
    mode: str
    level: int | None
    model_version: str
    practice: PracticeSession
    created_at: float
    last_seen: float
    # Cada conexión nueva (reconexión tras segundo plano) reemplaza a la anterior.
    connection: int = 0
    socket: WebSocket | None = None


class SessionStore:
    def __init__(
        self, ttl_seconds: float = SESSION_TTL_SECONDS, clock: Callable[[], float] = time.monotonic
    ):
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._sessions: dict[str, StoredSession] = {}

    def create(
        self, mode: str, level: int | None, model_version: str, practice: PracticeSession
    ) -> StoredSession:
        self._expire()
        now = self.clock()
        session_id = f"sess_{secrets.token_hex(8)}"
        session = StoredSession(session_id, mode, level, model_version, practice, now, now)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> StoredSession | None:
        self._expire()
        return self._sessions.get(session_id)

    def touch(self, session: StoredSession) -> None:
        session.last_seen = self.clock()

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _expire(self) -> None:
        now = self.clock()
        for session_id in [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_seen > self.ttl_seconds
        ]:
            del self._sessions[session_id]


_store = SessionStore()


def get_store() -> SessionStore:
    return _store
