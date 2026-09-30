"""El guante de la API: la laptop lo lee por BLE y cada sesión toma sus lecturas recientes.

Con `GLOVE_LIVE=true` se conecta al arrancar la API y se reconecta solo si se cae.
"""

from __future__ import annotations

from app.core.config import settings
from app.vision.glove import GloveFeed

_feed: GloveFeed | None = None


def start_glove_feed() -> GloveFeed | None:
    global _feed
    if not settings.glove_live:
        print("Guante BLE desactivado (GLOVE_LIVE=false); no se intentará conectar.", flush=True)
        return None
    if _feed is None:
        print("GLOVE_LIVE=true; iniciando lector BLE del guante.", flush=True)
        _feed = GloveFeed(settings.glove_device_name, settings.glove_characteristic_uuid)
        _feed.start()
    return _feed


def stop_glove_feed() -> None:
    global _feed
    if _feed is not None:
        _feed.close()
        _feed = None


def get_glove_feed() -> GloveFeed | None:
    return _feed
