from functools import lru_cache
from pathlib import Path

from pydantic import Field, NonNegativeInt, PositiveInt
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
    vision_grace_ms: NonNegativeInt = 300  # un tropiezo más corto no reinicia el sostener
    vision_wrong_sign_ms: NonNegativeInt = 500  # otra seña sostenida esto antes de avisar

    # Guante BLE: valores del receptor ESP32 de `manitas/config/settings.py`.
    glove_device_name: str = "GuanteLSM"
    glove_characteristic_uuid: str = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
    # API: la laptop lee el guante por BLE y lo junta con cada observación del iPhone.
    glove_live: bool = False  # conectar el guante al arrancar la API
    glove_required: bool = False  # sin lecturas recientes la práctica no evalúa (`disconnected`)
    # Hay un guante por mano; cada paquete BLE dice de cuál viene ("emisor").
    glove_right_emitter: PositiveInt = 1  # guante de la mano derecha
    glove_left_emitter: PositiveInt = 2  # guante de la mano izquierda
    glove_max_age_ms: PositiveInt = 500  # lecturas más viejas cuentan como guante desconectado
    glove_tilt_tolerance_deg: PositiveInt = 30  # desviación de inclinación permitida
    # Imprime en la consola del servidor las dos manos y los dos guantes de cada sesión.
    input_log: bool = False
    # false: las correcciones de la cámara solo se muestran, no hacen fallar el intento.
    feedback_camera_blocks: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
