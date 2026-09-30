"""`POST /sessions` y el WebSocket de cada sesión (RF-12, RF-14): contrato móvil 0.2.0."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.api.errors import ApiError
from app.catalog import get_catalog
from app.core.config import settings
from app.sessions.glove import get_glove_feed
from app.sessions.practice import ObservationError, PracticeSession
from app.sessions.protocol import (
    PROTOCOL_VERSION,
    ClientMessage,
    CreateSessionIn,
    CreateSessionOut,
    EndSessionIn,
    supported_version,
)
from app.sessions.store import (
    ModelNotAvailable,
    Models,
    SessionStore,
    get_store,
    load_models,
)
from app.vision.glove import GloveFeed

router = APIRouter(tags=["sessions"])
logger = logging.getLogger(__name__)

SESSION_NOT_FOUND_CLOSE = 4404
REPLACED_CLOSE = 4000
MIN_SAMPLE_RATE_HZ = 15

ModelLoader = Callable[[], Awaitable[Models]]


def get_model_loader() -> ModelLoader:
    async def load() -> Models:
        return await run_in_threadpool(load_models)

    return load


Store = Annotated[SessionStore, Depends(get_store)]
Loader = Annotated[ModelLoader, Depends(get_model_loader)]
Glove = Annotated[GloveFeed | None, Depends(get_glove_feed)]


def websocket_path(session_id: str) -> str:
    return f"{settings.api_v1_prefix}/ws/sessions/{session_id}"


@router.post("/sessions", status_code=201)
async def create_session(
    request: CreateSessionIn, store: Store, load: Loader, glove: Glove
) -> CreateSessionOut:
    if not supported_version(request.client_version):
        raise ApiError(
            426,
            "PROTOCOL_VERSION_UNSUPPORTED",
            f"La app usa el protocolo {request.client_version}; el servidor, {PROTOCOL_VERSION}",
        )
    catalog = get_catalog()
    target = None
    if request.mode == "practice":
        if request.target_sign is None:
            raise ApiError(400, "INVALID_REQUEST", "target_sign es obligatorio en modo practice")
        target = catalog.get(request.target_sign)
        if target is None:
            raise ApiError(
                404,
                "SIGN_NOT_FOUND",
                f"La seña {request.target_sign!r} no está en el catálogo",
                {"target_sign": request.target_sign},
            )

    try:
        models = await load()
    except ModelNotAvailable as error:
        raise ApiError(503, "MODEL_NOT_AVAILABLE", str(error)) from None
    recognizer = models.new_recognizer(with_movement=target is None or target.type != "static")
    if target is not None and target.data_label not in recognizer.signs:
        raise ApiError(
            503,
            "SIGN_NOT_TRAINED",
            f"El modelo todavía no reconoce {target.display_name}: faltan datos de esa seña",
            {"target_sign": target.id},
        )

    session = store.create(
        request.mode,
        target.level if target else None,
        models.version,
        PracticeSession(recognizer, catalog, target, glove.recent if glove else None),
    )
    return CreateSessionOut(
        session_id=session.id,
        mode=request.mode,
        target_sign=target.id if target else None,
        level=session.level,
        websocket_path=websocket_path(session.id),
        expires_in_seconds=int(store.ttl_seconds),
        catalog_version=catalog.version,
        model_version=models.version,
    )


@router.websocket("/ws/sessions/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str, store: Store) -> None:
    # Se acepta antes de cerrar: si no, el cliente recibe un 403 en vez del código 4404.
    await websocket.accept()
    session = store.get(session_id)
    if session is None:
        await websocket.close(SESSION_NOT_FOUND_CLOSE, "SESSION_NOT_FOUND")
        return

    # Al volver de segundo plano la app se reconecta a la misma sesión: gana la conexión nueva.
    previous = session.socket
    session.connection += 1
    connection, session.socket = session.connection, websocket
    if previous is not None:
        # Suele estar medio muerta (el iPhone la suspendió): cualquier error al cerrarla se ignora.
        with contextlib.suppress(Exception):
            await previous.close(REPLACED_CLOSE, "replaced")
    store.touch(session)

    await websocket.send_json(
        {
            "type": "ready",
            "session_id": session.id,
            "protocol_version": PROTOCOL_VERSION,
            "model_version": session.model_version,
            "min_sample_rate_hz": MIN_SAMPLE_RATE_HZ,
            # El cuerpo se usa en todos los niveles para validar el encuadre.
            "required_inputs": ["hand", "pose"]
            + (["glove"] if session.practice.glove_required else []),
            "server_timestamp_ms": round((store.clock() - session.created_at) * 1000),
        }
    )

    try:
        while True:
            raw = await websocket.receive_text()
            if session.connection != connection:
                return
            store.touch(session)
            try:
                message = ClientMessage.validate_json(raw)
            except ValidationError as error:
                await websocket.send_json(_invalid(raw, error))
                continue

            if isinstance(message, EndSessionIn):
                await websocket.send_json(session.practice.summary(session.id))
                store.remove(session.id)
                await websocket.close(1000, message.reason)
                return

            try:
                feedback = session.practice.observe(message)
            except ObservationError as error:
                feedback = _error(message.sequence, "INVALID_OBSERVATION", str(error))
            except Exception:
                logger.exception("Error al evaluar la observación %s", message.sequence)
                feedback = _error(
                    message.sequence, "INFERENCE_ERROR", "Error interno al evaluar la seña"
                )
            await websocket.send_json(feedback)
    except WebSocketDisconnect:
        pass
    finally:
        if session.connection == connection:
            session.socket = None


def _error(sequence: int | None, code: str, message: str) -> dict:
    return {"type": "error", "sequence": sequence, "code": code, "message": message}


def _invalid(raw: str, error: ValidationError) -> dict:
    try:
        data = json.loads(raw)
    except ValueError:
        return _error(None, "INVALID_REQUEST", "El mensaje no es JSON válido")
    sequence = data.get("sequence") if isinstance(data, dict) else None
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"])
    return _error(
        sequence if isinstance(sequence, int) else None,
        "INVALID_OBSERVATION",
        f"{field}: {first['msg']}" if field else first["msg"],
    )
