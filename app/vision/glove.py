"""Lectura BLE del guante para capturas de entrenamiento combinadas."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

GLOVE_VALUES = [
    "izq",
    "der",
    "arr",
    "abj",
    "giro_izq",
    "giro_der",
    "pulgar",
    "indice",
    "medio",
    "anular",
    "menique",
]
FINGERS = ("pulgar", "indice", "medio", "anular", "menique")
EMITTERS = (1, 2)
MIN_GLOVE_PACKETS = 5
MAX_LINE_BYTES = 1024
FEED_HISTORY = 256  # lecturas que guarda GloveFeed (~8 s a 30 paquetes/s)


class GloveCollector:
    """Conecta al receptor BLE en un hilo y conserva paquetes durante una práctica."""

    def __init__(self, device_name: str, characteristic_uuid: str, verbose: bool = False):
        self.device_name = device_name
        self.characteristic_uuid = characteristic_uuid
        self.verbose = verbose
        self.packets: list[tuple[int, int, list[float]]] = []
        self.dropped = 0
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("El lector del guante ya está iniciado")
        self._thread = threading.Thread(target=self._thread_main, name="glove-ble", daemon=True)
        self._thread.start()

    def mark(self) -> int:
        with self._lock:
            return len(self.packets)

    def since(self, mark: int) -> list[tuple[int, int, list[float]]]:
        with self._lock:
            return list(self.packets[mark:])

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as error:  # el error se muestra en la ventana principal al guardar
            self.error = error

    async def _run(self) -> None:
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError as error:
            raise RuntimeError(
                "Falta la dependencia BLE: instala el proyecto con `uv sync`"
            ) from error

        while not self._stop.is_set():
            print(f"Buscando guante BLE '{self.device_name}'...")
            device = await BleakScanner.find_device_by_name(self.device_name, timeout=10.0)
            if device is None:
                print(f"No se encontró '{self.device_name}'; reintentando...")
                await asyncio.sleep(2)
                continue
            try:
                async with BleakClient(device.address, timeout=15.0) as client:
                    self._buffer.clear()
                    await client.start_notify(self.characteristic_uuid, self._on_notification)
                    print("Conexión BLE del guante activa.")
                    while not self._stop.is_set() and client.is_connected:
                        if await asyncio.to_thread(self._stop.wait, 0.25):
                            break
            except Exception as error:
                self.error = error
                print(f"Error BLE ({type(error).__name__}); reintentando...")
                await asyncio.sleep(3)

    def _on_notification(self, _sender, data: bytearray) -> None:
        self._buffer.extend(data)
        while b"\n" in self._buffer:
            end = self._buffer.index(b"\n")
            line = bytes(self._buffer[:end]).strip()
            del self._buffer[: end + 1]
            if line:
                self._handle_line(line)
        if len(self._buffer) > MAX_LINE_BYTES:
            self._buffer.clear()
            self.dropped += 1

    def _handle_line(self, line: bytes) -> None:
        try:
            packet = json.loads(line.decode("utf-8"))
            emitter, values = packet["emisor"], packet["valores"]
            if emitter not in EMITTERS or len(values) != len(GLOVE_VALUES):
                raise ValueError("emisor o cantidad de valores inválidos")
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
            ):
                raise ValueError("valores no numéricos")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.dropped += 1
            return
        packet = (int(time.time() * 1000), int(emitter), [float(value) for value in values])
        self._store(packet)
        if self.verbose:
            print(f"Guante E{emitter}: {values}")

    def _store(self, packet: tuple[int, int, list[float]]) -> None:
        with self._lock:
            self.packets.append(packet)


@dataclass(frozen=True)
class GloveReading:
    """Una lectura del guante con nombre.

    Dedos: 1 = encogido, 2 = a medias, 3 = estirado. El MPU manda cada eje partido en dos
    valores positivos (grados): `izq`/`der`, `arr`/`abj` y `giro_izq`/`giro_der`; uno de los dos
    vale 0. Aquí se juntan en un ángulo con signo por eje.
    """

    t_ms: float
    fingers: dict[str, float]
    roll: float  # der - izq: inclinación hacia los lados
    pitch: float  # arr - abj: inclinación hacia arriba/abajo
    yaw: float  # giro_der - giro_izq: se deriva con el tiempo; no se usa para corregir

    @classmethod
    def from_values(cls, t_ms: float, values: list[float]) -> GloveReading:
        named = dict(zip(GLOVE_VALUES, values, strict=True))
        return cls(
            t_ms=t_ms,
            fingers={finger: named[finger] for finger in FINGERS},
            roll=named["der"] - named["izq"],
            pitch=named["arr"] - named["abj"],
            yaw=named["giro_der"] - named["giro_izq"],
        )


class GloveFeed(GloveCollector):
    """Lector BLE para la API: guarda solo las lecturas recientes de un emisor.

    Los paquetes llevan la hora de la laptop al llegar (no la del iPhone): cada observación
    toma las lecturas de los últimos milisegundos según el reloj del servidor.
    """

    def __init__(
        self,
        device_name: str,
        characteristic_uuid: str,
        emitter: int = 1,
        history: int = FEED_HISTORY,
    ):
        super().__init__(device_name, characteristic_uuid)
        self.emitter = emitter
        self._recent: deque[GloveReading] = deque(maxlen=history)

    def _store(self, packet: tuple[int, int, list[float]]) -> None:
        t_ms, emitter, values = packet
        if emitter != self.emitter:
            return
        with self._lock:
            self._recent.append(GloveReading.from_values(t_ms, values))

    def recent(self, window_ms: float, now_ms: float | None = None) -> list[GloveReading]:
        """Lecturas de los últimos `window_ms`; vacío si el guante no manda nada."""
        now_ms = time.time() * 1000 if now_ms is None else now_ms
        with self._lock:
            return [reading for reading in self._recent if now_ms - reading.t_ms <= window_ms]


def write_packets(path: Path, packets: list[tuple[int, int, list[float]]]) -> None:
    """Escribe el formato crudo compatible con el proyecto `manitas`."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["t_ms", "emisor", *GLOVE_VALUES])
        for t_ms, emitter, values in packets:
            writer.writerow([t_ms, emitter, *values])
