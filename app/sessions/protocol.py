"""Mensajes del contrato móvil (docs/mobile-api-contract.md) que llegan de la app."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, TypeAdapter

PROTOCOL_VERSION = "0.2.0"
MAX_HANDS = 2

Point3 = tuple[float, float, float]
Point4 = tuple[float, float, float, float]


class CreateSessionIn(BaseModel):
    mode: Literal["practice", "demo"] = "practice"
    target_sign: str | None = None
    participant_id: str | None = None
    device_id: str | None = None
    client_version: str
    calibration_id: str | None = None
    # Aún no se guarda nada aunque venga en true: falta definir la política (contrato).
    record: bool = False


class CreateSessionOut(BaseModel):
    session_id: str
    mode: Literal["practice", "demo"]
    target_sign: str | None
    level: int | None
    status: Literal["created"] = "created"
    websocket_path: str
    expires_in_seconds: int
    protocol_version: str = PROTOCOL_VERSION
    catalog_version: str
    model_version: str


class Handedness(BaseModel):
    label: str
    score: float


class HandIn(BaseModel):
    landmarks: Annotated[list[Point3], Field(min_length=21, max_length=21)]
    handedness: Handedness


class VisionIn(BaseModel):
    image_width: PositiveInt
    image_height: PositiveInt
    mirrored: bool
    hands: Annotated[list[HandIn], Field(max_length=MAX_HANDS)]
    pose_landmarks: Annotated[list[Point4], Field(min_length=33, max_length=33)] | None = None


class ObservationIn(BaseModel):
    type: Literal["observation"]
    sequence: NonNegativeInt
    timestamp_ms: Annotated[float, Field(ge=0)]
    vision: VisionIn
    glove: dict | None = None


class EndSessionIn(BaseModel):
    type: Literal["end_session"]
    reason: str = "user_finished"


ClientMessage = TypeAdapter(Annotated[ObservationIn | EndSessionIn, Field(discriminator="type")])


def supported_version(client_version: str) -> bool:
    """Mismo major.minor que el protocolo: los cambios de parche son compatibles."""
    return client_version.split(".")[:2] == PROTOCOL_VERSION.split(".")[:2]
