from functools import lru_cache
from pathlib import Path

from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ruta absoluta: la CLI `vision` también lee el .env aunque se ejecute desde otra carpeta.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Mecabite API"
    environment: str = "local"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Reconocimiento (app/vision/recognizer.py). Se lee al arrancar la API o la CLI.
    vision_hold_ms: PositiveInt = 1000  # sostener una estática para confirmarla (RF-06)
    vision_confidence_threshold: float = Field(0.65, ge=0, le=1)
    vision_vote_ms: PositiveInt = 400  # ventana de votos del modelo estático

    # Guante BLE: valores del receptor ESP32 de `manitas/config/settings.py`.
    glove_device_name: str = "GuanteLSM"
    glove_characteristic_uuid: str = "beb5483e-36e1-4688-b7f5-ea07361b26a8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
