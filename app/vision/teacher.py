"""Recolección, entrenamiento y prueba en vivo del clasificador de visión.

Ejemplos (`vision` equivale a `python -m app.vision`):
    vision collect --label A --participant p01
    vision collect --label reposo --participant p01
    vision train
    vision demo
    vision live --target A
    vision practice --participant omar      # pide seña por seña y guarda las reconocidas
    vision clear                            # borra todo lo de practice
    vision import-videos --source data/raw/MSL-dynamic-signs

`collect` solo guarda landmarks y timestamps. `practice` guarda además una foto (estáticas)
o un clip corto (con movimiento) en data/practice/, que queda fuera de Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import unicodedata
from collections import Counter, deque
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import GroupKFold

from app.catalog import load_catalog as load_sign_catalog
from app.core.config import settings
from app.vision.body import BodyStatus, body_status
from app.vision.catalog import OTHER, Catalog, load_catalog
from app.vision.classifier import BACKENDS, SignClassifier, make_backend
from app.vision.features import NUM_LANDMARKS, Hand, landmarks_to_vector
from app.vision.glove import GLOVE_VALUES, MIN_GLOVE_PACKETS, GloveCollector, write_packets
from app.vision.recognizer import (
    CONFIDENCE_THRESHOLD,
    DYNAMIC_MODEL_PATH,
    HOLD_SECONDS,
    STATIC_MODEL_PATH,
    Recognizer,
    alphabet_key,
)
from app.vision.sequence import (
    CONFIDENCE_THRESHOLD as DYNAMIC_THRESHOLD,
)
from app.vision.sequence import (
    END_STILL_MS,
    MOVING_SPEED,
    SPEED_LAG_MS,
    augment,
    hand_speed,
    held_pose,
    load_sequence,
    sequence_features,
    transition,
)
from app.vision.tracker import (
    NUM_BODY_LANDMARKS,
    BodyTracker,
    HandSlots,
    HandTracker,
    draw_body,
    draw_hand,
    open_camera,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "vision"
RAW_DIR = ROOT_DIR / "data" / "raw"
DYNAMIC_VARIANTS = 3  # variaciones de velocidad/pausas por grabación
NEGATIVE_SAMPLES = 1500  # de cada tipo: forma quieta y cambio de forma
# Grabaciones propias (collect/practice): son pocas frente a los datasets, pero son justo la
# cámara y la forma de hacer las señas que importan en vivo; cuentan este número de veces.
OWN_WEIGHT = 5
# Personas de los datasets: S (MSL-ABC y MSL-dynamic-signs), G (glosas de Zenodo) y
# M (249 palabras de Mendeley). Cualquier otro nombre es una grabación propia.
DATASET_PERSON = re.compile(r"[SGM]\d+")
VIDEO_NAME = re.compile(r"^(S\d+)-([^-]+)-", re.IGNORECASE)
PERSON_ID = re.compile(r"(?:^|_)([SGM]\d+)(?:-|$)")
GLOSS_VIDEO = re.compile(r"^(.+)_(\d+)$")  # HOLA_3.mp4: glosa y persona (desde 0)
# Pocas grabaciones por seña (palabras): se generan más variaciones para que la clase pese
# como las letras, que traen cientos.
MIN_CLASS_RECORDINGS = 100
HEADER = ["t_ms", "is_left"] + [
    f"{axis}{index}" for index in range(NUM_LANDMARKS) for axis in "xyz"
]
# collect/practice agregan el cuerpo (x, y, z, visibilidad); vacío si no se detectó.
BODY_HEADER = [
    f"body_{axis}{index}" for index in range(NUM_BODY_LANDMARKS) for axis in ("x", "y", "z", "v")
]
# La otra mano (señas de dos manos, como gracias); vacía si no se vio.
OTHER_HEADER = ["other_is_left"] + [
    f"other_{axis}{index}" for index in range(NUM_LANDMARKS) for axis in "xyz"
]
RECORD_HEADER = HEADER + BODY_HEADER + OTHER_HEADER
SECOND_HAND_COLOR = (230, 160, 60)  # la otra mano se dibuja en azul
SIGN_BOX = 170
# Datos de `vision practice`: carpeta propia para que `vision clear` nunca toque los datasets.
PRACTICE_DIR = ROOT_DIR / "data" / "practice"
PRACTICE_LOG = "sesiones.csv"
PRACTICE_TAG = "__practica_"  # <persona>__practica_<t_ms>: la persona queda como grupo al evaluar
CLIP_WIDTH = 640  # los clips de señas con movimiento se guardan reducidos
SAVED_MESSAGE_SECONDS = 1.2
PRACTICE_WINDOW = "Mecabite - practica"
GLOVE_HOLD_MS = 1000
GLOVE_CAPTURE_WINDOW_MS = 1000
MOVEMENT_CLIP_MS = 2000
LEVEL_3_WORD_CLIP_MS = 4000
LEVEL_3_DETECTION_TIMEOUT_MS = 2000
CLIP_HISTORY_MS = LEVEL_3_WORD_CLIP_MS + 500
GLOVE_REVIEW_SAMPLE_MS = 200
GLOVE_REVIEW_PAGE_SIZE = 10


def _draw_text(frame: np.ndarray, text: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _record_row(
    t_ms: int,
    points: np.ndarray,
    is_left: bool,
    body: np.ndarray | None,
    other: Hand | None = None,
) -> list[object]:
    body_values = np.round(body.flatten(), 6).tolist() if body is not None else []
    body_values += [""] * (len(BODY_HEADER) - len(body_values))
    other_values: list[object] = [""] * len(OTHER_HEADER)
    if other is not None:
        other_values = [int(other[1]), *np.round(other[0].flatten(), 6)]
    return [t_ms, int(is_left), *np.round(points.flatten(), 6), *body_values, *other_values]


class BodyView:
    """Cuerpo en vivo: detecta los puntos, los dibuja y resume encuadre y localización."""

    def __init__(self, enabled: bool = True):
        self.tracker = BodyTracker() if enabled else None
        self.body: np.ndarray | None = None
        self.status: BodyStatus | None = None

    @property
    def framed(self) -> bool:
        return self.tracker is None or (self.status is not None and self.status.framed)

    def update(self, frame: np.ndarray, timestamp: int, hand: np.ndarray | None) -> None:
        """Llamar con el cuadro limpio, antes de dibujar encima."""
        if self.tracker is None:
            return
        self.body = self.tracker.detect(frame, timestamp)
        self.status = body_status(hand, self.body, frame.shape[1] / frame.shape[0])

    def draw(self, frame: np.ndarray) -> None:
        if self.tracker is None or self.status is None:
            return
        if self.body is not None:
            draw_body(frame, self.body)
        if self.status.issue:
            _draw_text(frame, f"Encuadre: {self.status.issue}", 146, (0, 200, 255))
        else:
            where = self.status.location or "-"
            _draw_text(frame, f"Encuadre OK | mano en: {where}", 146, (80, 220, 120))

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()


def collect(label: str, participant: str, camera: int, body_enabled: bool = True) -> None:
    output_dir = DATA_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker, capture = HandTracker(), open_camera(camera)
    body_view = BodyView(body_enabled)
    file_handle = None
    writer = None
    frames = 0
    recording_path: Path | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            timestamp = int(time.time() * 1000)
            detection = tracker.detect(frame, timestamp)
            body_view.update(frame, timestamp, detection[0] if detection else None)

            if detection:
                points, is_left = detection
                draw_hand(frame, points)
                if tracker.other:
                    draw_hand(frame, tracker.other[0], SECOND_HAND_COLOR)
                if writer:
                    row = _record_row(timestamp, points, is_left, body_view.body, tracker.other)
                    writer.writerow(row)
                    frames += 1
            body_view.draw(frame)

            if writer:
                _draw_text(
                    frame,
                    f"GRABANDO {label}: {frames} cuadros | ESPACIO termina",
                    32,
                    (60, 60, 255),
                )
            else:
                _draw_text(frame, f"{label} | ESPACIO graba | Q sale", 32)
            if not detection:
                _draw_text(frame, "No se detecta una mano", 64, (0, 200, 255))

            cv2.imshow("Mecabite - recoleccion de vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if writer:
                    file_handle.close()
                    print(f"Guardada {recording_path} con {frames} cuadros")
                    file_handle, writer, recording_path, frames = None, None, None, 0
                else:
                    recording_path = output_dir / f"{participant}_{int(time.time() * 1000)}.csv"
                    file_handle = recording_path.open("w", newline="")
                    writer = csv.writer(file_handle)
                    writer.writerow(RECORD_HEADER)
                    frames = 0
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if file_handle:
            file_handle.close()
        capture.release()
        cv2.destroyAllWindows()
        tracker.close()
        body_view.close()


def _parse_video_name(path: Path) -> tuple[str, str] | None:
    """`S21-Ñ-frontal-3.mp4` -> ("S21", "Ñ")."""
    match = VIDEO_NAME.match(unicodedata.normalize("NFC", path.stem))
    if not match:
        return None
    return match.group(1).upper(), match.group(2).upper()


def _video_import_identity(
    video_path: Path, label: str | None, participant: str
) -> tuple[str, str] | None:
    """Devuelve la etiqueta y el nombre del CSV para un video de importación."""
    if label:
        normalized_label = unicodedata.normalize("NFC", label)
        stem = unicodedata.normalize("NFC", video_path.stem).replace(" ", "_")
        return normalized_label, f"{participant}_{normalized_label}_{stem}"
    parsed = _parse_video_name(video_path)
    if parsed is None:
        return None
    _, sign = parsed
    stem = unicodedata.normalize("NFC", video_path.stem).replace(" ", "_")
    return sign, stem


def import_videos(
    source: Path,
    stride: int,
    overwrite: bool,
    label: str | None = None,
    participant: str = "p01",
) -> None:
    videos = sorted(source.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"No se encontraron videos .mp4 en {source}")
    print(f"Procesando {len(videos)} videos de {source}")

    stats = Counter()
    for number, video_path in enumerate(videos, start=1):
        identity = _video_import_identity(video_path, label, participant)
        if identity is None:
            print(f"  [{number}/{len(videos)}] se omite (nombre no reconocido): {video_path.name}")
            stats["omitidos"] += 1
            continue
        sign, stem = identity
        output_path = DATA_DIR / sign / f"{stem}.csv"
        if output_path.exists() and not overwrite:
            stats["existentes"] += 1
            continue

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        tracker = HandTracker()
        rows = []
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    # Se voltea para que coincida con la vista espejo de la webcam en collect/live.
                    frame = cv2.flip(frame, 1)
                    timestamp = int(index * 1000 / fps)
                    detection = tracker.detect(frame, timestamp)
                    if detection:
                        points, is_left = detection
                        rows.append([timestamp, int(is_left), *np.round(points.flatten(), 6)])
                index += 1
        finally:
            capture.release()
            tracker.close()

        if not rows:
            print(f"  [{number}/{len(videos)}] sin manos detectadas: {video_path.name}")
            stats["sin_manos"] += 1
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(HEADER)
            writer.writerows(rows)
        stats["importados"] += 1
        stats["cuadros"] += len(rows)
        print(f"  [{number}/{len(videos)}] {sign}: {len(rows)} cuadros <- {video_path.name}")

    print("Resumen: " + ", ".join(f"{key}={value}" for key, value in stats.items()))


def _video_frames(path: Path, target_fps: float = 30.0) -> Iterator[tuple[int, np.ndarray]]:
    """Cuadros de un video con su tiempo en ms, a lo más `target_fps` por segundo."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
    stride = max(1, round(fps / target_fps))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield int(index * 1000 / fps), frame
            index += 1
    finally:
        capture.release()


def _image_frames(folder: Path, fps: float) -> Iterator[tuple[int, np.ndarray]]:
    """Secuencia de fotos (JPG/PNG, en orden de nombre) como si fueran cuadros a `fps`."""
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for index, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is not None:
            yield int(index * 1000 / fps), frame


# (t_ms, puntos, mano izquierda, cuerpo, la otra mano)
Row = tuple[int, np.ndarray, bool, np.ndarray | None, Hand | None]


def _hand_coverage(hands: list[Hand | None]) -> float:
    """Área cubierta por la mano, ignorando el 10 % exterior de las observaciones.

    El recorrido acumulado puede favorecer una mano quieta con ruido o una mano que cruza varias
    veces el mismo sitio. La caja de todos sus landmarks mide el espacio realmente ocupado y
    sigue funcionando aunque MediaPipe pierda cuadros durante el movimiento.
    """
    observed = [hand[0][:, :2] for hand in hands if hand is not None]
    if not observed:
        return 0.0
    landmarks = np.concatenate(observed, axis=0)
    low, high = np.percentile(landmarks, [10, 90], axis=0)
    return float(np.prod(high - low))


def _track_sequence(frames: Iterator[tuple[int, np.ndarray]]) -> list[Row]:
    """Manos y cuerpo de cada cuadro, volteado como la webcam. Solo cuadros con mano.

    Se siguen las dos manos y queda como principal la que más se mueve: en gracias una mano
    se queda quieta de base y la seña la hace la otra."""
    tracker, body, slots = HandTracker(), BodyTracker(), HandSlots()
    tracked = []
    try:
        for t_ms, frame in frames:
            frame = cv2.flip(frame, 1)
            hands = [hand[:2] for hand in tracker.detect_hands(frame, t_ms)]
            tracked.append((t_ms, slots.assign(hands, t_ms), body.detect(frame, t_ms)))
    finally:
        tracker.close()
        body.close()
    active = max((0, 1), key=lambda slot: _hand_coverage([hands[slot] for _, hands, _ in tracked]))
    return [
        (t_ms, *hands[active], pose, hands[1 - active])
        for t_ms, hands, pose in tracked
        if hands[active] is not None
    ]


def _trim_to_motion(rows: list[Row]) -> list[Row]:
    """Quita la quietud antes y después de la seña, como la ve el detector en vivo: desde medio
    segundo antes de que la mano empiece a moverse hasta que se queda quieta."""
    if len(rows) < 3:
        return rows
    times = np.array([row[0] for row in rows], dtype=float)
    moving = []
    for index, (t_ms, points, *_) in enumerate(rows):
        earlier = int(np.searchsorted(times, t_ms - SPEED_LAG_MS, side="right")) - 1
        speed = 0.0
        if 0 <= earlier < index:
            speed = hand_speed(rows[earlier][1], points, t_ms - times[earlier])
        moving.append(speed >= MOVING_SPEED)
    indices = np.flatnonzero(moving)
    if not len(indices):
        return rows
    start, end = times[indices[0]] - 500, times[indices[-1]] + END_STILL_MS + 50
    return [row for row in rows if start <= row[0] <= end]


def _write_record(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(RECORD_HEADER)
        writer.writerows(_record_row(*row) for row in rows)


def import_glosses(source: Path, overwrite: bool) -> None:
    """Dataset de glosas de Zenodo: `<GLOSA>/<GLOSA>_<persona>.mp4`. Solo importa las glosas
    que están en el catálogo de la app (HOLA -> hola, POR_FAVOR -> por_favor, ...)."""
    signs = load_sign_catalog()
    videos = sorted(source.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"No se encontraron videos .mp4 en {source}")
    stats = Counter()
    for number, video_path in enumerate(videos, start=1):
        match = GLOSS_VIDEO.match(unicodedata.normalize("NFC", video_path.stem))
        sign = signs.find(match.group(1)) if match else None
        if sign is None:
            stats["fuera_del_catalogo"] += 1
            continue
        person = f"G{int(match.group(2)) + 1:02d}"
        label = sign.data_label
        output_path = DATA_DIR / label / f"{person}-{label}-glosas.csv"
        if output_path.exists() and not overwrite:
            stats["existentes"] += 1
            continue
        rows = _trim_to_motion(_track_sequence(_video_frames(video_path)))
        if not rows:
            print(f"  [{number}/{len(videos)}] sin manos detectadas: {video_path.name}")
            stats["sin_manos"] += 1
            continue
        _write_record(output_path, rows)
        stats["importados"] += 1
        seconds = (rows[-1][0] - rows[0][0]) / 1000
        print(f"  [{number}/{len(videos)}] {label}: {len(rows)} cuadros, {seconds:.1f} s")
    print("Resumen: " + ", ".join(f"{key}={value}" for key, value in stats.items()))


def import_frames(source: Path, label: str, fps: float, prefix: str, overwrite: bool) -> None:
    """Secuencias de fotos: una subcarpeta por persona (ej. Mendeley 249 palabras). Las
    personas quedan como `<prefix>01`, `<prefix>02`...; `fps` es el ritmo supuesto de las fotos."""
    folders = sorted(folder for folder in source.iterdir() if folder.is_dir())
    if not folders:
        raise SystemExit(f"No hay subcarpetas (una por persona) en {source}")
    label = unicodedata.normalize("NFC", label)
    stats = Counter()
    for number, folder in enumerate(folders, start=1):
        person = f"{prefix}{number:02d}"
        output_path = DATA_DIR / label / f"{person}-{label}-{source.name}.csv"
        if output_path.exists() and not overwrite:
            stats["existentes"] += 1
            continue
        rows = _trim_to_motion(_track_sequence(_image_frames(folder, fps)))
        if len(rows) < 3:
            print(f"  [{number}/{len(folders)}] muy pocos cuadros con mano: {folder.name}")
            stats["sin_manos"] += 1
            continue
        _write_record(output_path, rows)
        stats["importados"] += 1
        print(f"  [{number}/{len(folders)}] {label}: {len(rows)} cuadros <- {folder.name}")
    print("Resumen: " + ", ".join(f"{key}={value}" for key, value in stats.items()))


def _person(stem: str) -> str:
    """`msl-abc__S3`, `S3-J-frontal-1` -> "S3"; `G05-hola-glosas` -> "G05";
    `p01_1759...` -> "p01"."""
    match = PERSON_ID.search(stem)
    return match.group(1) if match else stem.split("_")[0]


def _is_own(person: str) -> bool:
    return not DATASET_PERSON.fullmatch(person)


def _label_dirs() -> list[tuple[str, Path]]:
    """Carpetas por seña de los datasets y de las sesiones de `vision practice`."""
    roots = (DATA_DIR, PRACTICE_DIR / "landmarks")
    return [
        (unicodedata.normalize("NFC", label_dir.name), label_dir)
        for root in roots
        for label_dir in sorted(root.glob("*/"))
    ]


def load_static_dataset(
    catalog: Catalog | None = None, mirror: bool = True
) -> tuple[np.ndarray, list[str], list[str]]:
    """Cuadros sueltos para las señas estáticas del catálogo; lo demás se aprende como `otra`.

    Con `mirror`, cada cuadro se agrega también reflejado: la etiqueta izquierda/derecha de
    MediaPipe falla seguido y los datasets son casi todos de una sola mano, así el modelo
    reconoce ambas manos sin depender de esa etiqueta.
    """
    catalog = catalog or load_catalog()
    features, labels, groups = [], [], []
    for folder, label_dir in _label_dirs():
        if folder in catalog.dynamic_folders:
            continue
        label = catalog.static_class(folder)
        for path in sorted(label_dir.glob("*.csv")):
            person = _person(path.stem)
            repeat = OWN_WEIGHT if _is_own(person) else 1
            _, points_seq, is_left = load_sequence(path)
            for points in points_seq:
                for flipped in (False, True) if mirror else (False,):
                    vector = landmarks_to_vector(points, is_left=is_left != flipped)
                    features.extend([vector] * repeat)
                    labels.extend([label] * repeat)
                    groups.extend([person] * repeat)
    return np.asarray(features), labels, groups


def load_dynamic_dataset(
    catalog: Catalog | None = None, seed: int = 42
) -> tuple[np.ndarray, list[str], list[str]]:
    """Una ventana por grabación de seña con movimiento (con variaciones) más negativos:
    formas quietas, cambios de forma y señas con movimiento fuera del catálogo (`otra`)."""
    catalog = catalog or load_catalog()
    rng = np.random.default_rng(seed)
    features, labels, groups = [], [], []
    poses: dict[str, list[np.ndarray]] = {}

    def add(t_ms: np.ndarray, points: np.ndarray, is_left: bool, label: str, person: str) -> None:
        for flipped in (False, True):
            features.append(sequence_features(t_ms, points, is_left != flipped))
            labels.append(label)
            groups.append(person)

    def canonical(points: np.ndarray, is_left: bool) -> np.ndarray:
        # Formas de negativos siempre como mano derecha; `add` agrega la versión en espejo.
        mirrored = points.copy()
        if is_left:
            mirrored[:, 0] = -mirrored[:, 0]
        return mirrored

    recordings = Counter(
        folder for folder, label_dir in _label_dirs() for _ in label_dir.glob("*.csv")
    )
    for folder, label_dir in _label_dirs():
        dynamic = folder in catalog.dynamic_folders
        # Señas con pocas grabaciones (palabras): más variaciones para que pesen como las letras.
        scarce = max(1, round(MIN_CLASS_RECORDINGS / max(recordings[folder], 1)))
        for path in sorted(label_dir.glob("*.csv")):
            person = _person(path.stem)
            t_ms, points_seq, is_left = load_sequence(path)
            if dynamic and len(points_seq) >= 3:
                # Las propias cuentan más: más variaciones de velocidad y pausas.
                variants = DYNAMIC_VARIANTS * (OWN_WEIGHT if _is_own(person) else scarce)
                for _ in range(variants):
                    augmented = augment(t_ms, points_seq, rng)
                    add(*augmented, is_left, catalog.dynamic_class(folder), person)
                # La forma inicial y final quietas no son la letra.
                picks = [0, len(points_seq) - 1]
            elif not dynamic and len(points_seq):
                picks = rng.choice(len(points_seq), size=min(4, len(points_seq)), replace=False)
            else:
                continue
            poses.setdefault(person, []).extend(
                canonical(points_seq[index], is_left) for index in picks
            )

    people = [person for person, shapes in poses.items() if len(shapes) >= 2]
    if not people:
        return np.asarray(features), labels, groups
    for _ in range(NEGATIVE_SAMPLES):
        person = people[rng.integers(len(people))]
        shapes = poses[person]
        first, second = rng.choice(len(shapes), size=2, replace=False)
        add(*held_pose(shapes[first], rng), False, OTHER, person)
        add(*transition(shapes[first], shapes[second], rng), False, OTHER, person)
    return np.asarray(features), labels, groups


def evaluate_by_person(
    model_name: str,
    features: np.ndarray,
    labels: list[str],
    groups: list[str],
    threshold: float,
) -> None:
    """Validación cruzada dejando fuera personas completas (RNF-03: usuarios distintos).

    Una predicción con confianza menor al umbral cuenta como `otra`, igual que en vivo.
    """
    labels_array, groups_array = np.asarray(labels), np.asarray(groups)
    folds = min(5, len(set(groups)))
    if folds < 3:
        print("Aviso: hay muy pocas personas para evaluar con gente nueva.")
        return
    recalls: dict[str, list[float]] = {}
    false_positives = []
    own_hits: list[bool] = []  # señas propias, evaluadas sin haber visto a esa persona
    for train_idx, test_idx in GroupKFold(folds).split(features, labels_array, groups_array):
        classifier = SignClassifier(make_backend(model_name))
        classifier.train(features[train_idx], labels_array[train_idx].tolist())
        probabilities = classifier.backend.predict_proba(features[test_idx])
        predicted = np.asarray(classifier.labels)[probabilities.argmax(axis=1)]
        accepted = np.where(probabilities.max(axis=1) >= threshold, predicted, OTHER)
        truth = labels_array[test_idx]
        for label in sorted(set(truth) - {OTHER}):
            recalls.setdefault(label, []).append(float(np.mean(accepted[truth == label] == label)))
        if np.any(truth == OTHER):
            false_positives.append(float(np.mean(accepted[truth == OTHER] != OTHER)))
        own = np.array([_is_own(group) for group in groups_array[test_idx]]) & (truth != OTHER)
        own_hits.extend((accepted[own] == truth[own]).tolist())

    per_label = "  ".join(f"{label}={np.mean(values):.0%}" for label, values in recalls.items())
    average = np.mean([np.mean(values) for values in recalls.values()])
    print(f"  Personas nuevas ({folds} grupos): acierto {average:.1%} | {per_label}")
    if false_positives:
        print(f"  Falsos positivos (otra cosa tomada como seña): {np.mean(false_positives):.1%}")
    if own_hits:
        people = ", ".join(sorted({group for group in groups if _is_own(group)}))
        print(
            f"  Con tus grabaciones ({people}), sin haberlas visto: {np.mean(own_hits):.1%}"
            f" (mide qué tan bien funciona con tu cámara)"
        )


def train(model_name: str, only: str | None = None) -> None:
    catalog = load_catalog()
    if only in (None, "static"):
        print(f"Modelo estático (configuración, se valida sosteniendo {HOLD_SECONDS:g} s)")
        features, labels, groups = load_static_dataset(catalog)
        if len(set(labels)) < 2:
            raise SystemExit("Se necesitan al menos dos clases de señas estáticas.")
        print(f"  {len(features)} cuadros | " + _counts(labels))
        evaluate_by_person(model_name, features, labels, groups, CONFIDENCE_THRESHOLD)
        _fit_and_save(model_name, features, labels, STATIC_MODEL_PATH)

    if only in (None, "dynamic"):
        print("Modelo dinámico (trayectoria de los últimos 2 s)")
        features, labels, groups = load_dynamic_dataset(catalog)
        if len(set(labels)) < 2:
            print("  Sin datos de señas con movimiento; se omite.")
            return
        print(f"  {len(features)} ventanas | " + _counts(labels))
        evaluate_by_person(model_name, features, labels, groups, DYNAMIC_THRESHOLD)
        _fit_and_save(model_name, features, labels, DYNAMIC_MODEL_PATH)


def _counts(labels: list[str]) -> str:
    return ", ".join(f"{label}={count}" for label, count in sorted(Counter(labels).items()))


def _fit_and_save(model_name: str, features: np.ndarray, labels: list[str], path: Path) -> None:
    classifier = SignClassifier(make_backend(model_name))
    classifier.train(features, labels)
    classifier.save(path)
    print(f"  Guardado en {path.relative_to(ROOT_DIR)} ({path.stat().st_size / 1e6:.1f} MB)")


def _ascii(label: str) -> str:
    # Las fuentes de OpenCV solo tienen ASCII.
    return label.replace("Ñ", "N~")


def _draw_sign_box(
    frame: np.ndarray, sign: str | None, pending: str | None, progress: float
) -> None:
    """Recuadro en la esquina superior derecha con la seña confirmada."""
    width = frame.shape[1]
    x0, y0 = width - SIGN_BOX - 12, 12
    x1, y1 = x0 + SIGN_BOX, y0 + SIGN_BOX
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 220, 120) if sign else (150, 150, 150), 3)

    text = "N" if sign == "Ñ" else (sign or "-")
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 8
    (text_w, _), _ = cv2.getTextSize(text, font, 1.0, thickness)
    scale = min(4.0, SIGN_BOX * 0.7 / max(text_w, 1))
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = (x0 + (SIGN_BOX - text_w) // 2, y0 + (SIGN_BOX + text_h) // 2 + 8)
    cv2.putText(frame, text, origin, font, scale, (255, 255, 255), thickness)
    if sign == "Ñ":
        tilde_y = origin[1] - text_h - 14
        xs = np.linspace(origin[0], origin[0] + text_w, 20)
        ys = tilde_y + 7 * np.sin(np.linspace(0, 2 * np.pi, 20))
        points = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(frame, [points], False, (255, 255, 255), 6)

    if pending and progress > 0:
        bar_y = y1 + 8
        cv2.rectangle(frame, (x0, bar_y), (x1, bar_y + 10), (60, 60, 60), -1)
        cv2.rectangle(
            frame, (x0, bar_y), (x0 + int(SIGN_BOX * progress), bar_y + 10), (80, 220, 120), -1
        )
        label = f"{_ascii(pending)}..."
        cv2.putText(frame, label, (x0, bar_y + 34), font, 0.8, (0, 0, 0), 5)
        cv2.putText(frame, label, (x0, bar_y + 34), font, 0.8, (255, 255, 255), 2)


def _load_recognizer() -> Recognizer:
    try:
        return Recognizer.load()
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None


def _draw_recognizer(frame: np.ndarray, recognizer: Recognizer, t_ms: float) -> None:
    """Estado arriba a la izquierda y la seña confirmada en la esquina."""
    if recognizer.frame_label:
        label, confidence = recognizer.frame_label
        hand = {None: "?", True: "izq", False: "der"}[recognizer.is_left]
        _draw_text(frame, f"Cuadro: {_ascii(label)} ({confidence:.0%}) | mano {hand}", 32)
    else:
        _draw_text(frame, "No se detecta una mano", 32, (0, 200, 255))
    if recognizer.detector is not None:
        if recognizer.moving:
            _draw_text(frame, "Movimiento: siguiendo la trayectoria...", 108, (0, 200, 255))
        elif recognizer.last_movement:
            moving_label, moving_confidence = recognizer.last_movement
            _draw_text(
                frame,
                f"Ultimo movimiento: {_ascii(moving_label)} ({moving_confidence:.0%})",
                108,
            )
    stabilizer = recognizer.stabilizer
    _draw_sign_box(frame, stabilizer.confirmed, stabilizer.candidate, recognizer.progress(t_ms))


def _resolve_signs(names: list[str], known: list[str]) -> list[str]:
    """Nombres escritos por la persona -> clases del modelo, sin importar mayúsculas ni
    espacios: `hola`, `HOLA` -> hola; `por favor` -> por_favor; `ñ` -> Ñ."""

    def key(name: str) -> str:
        return unicodedata.normalize("NFC", name).strip().casefold().replace(" ", "_")

    by_key = {key(sign): sign for sign in known}
    unknown = [name for name in names if key(name) not in by_key]
    if unknown:
        raise SystemExit(
            f"El modelo no reconoce: {', '.join(unknown)}. Disponibles: {' '.join(known)}"
        )
    return [by_key[key(name)] for name in names]


def _draw_hands(frame: np.ndarray, detection: Hand | None, other: Hand | None) -> None:
    if detection:
        draw_hand(frame, detection[0])
    if other:
        draw_hand(frame, other[0], SECOND_HAND_COLOR)


def _two_hands_message(sign: str) -> str:
    return f"{_ascii(sign)} se hace con las dos manos: levanta tambien la otra"


def live(target: str | None, camera: int, body_enabled: bool = True) -> None:
    """Sin `target` funciona como demo: solo muestra la seña reconocida.

    Con `target`, un intento solo cuenta si además la persona está bien encuadrada (de
    frente, con cara y hombros a la vista): si no, se indica qué corregir (RF-13).
    """
    recognizer = _load_recognizer()
    if target is not None:
        target = _resolve_signs([target], recognizer.signs)[0]
    tracker, capture = HandTracker(), open_camera(camera)
    body_view = BodyView(body_enabled)
    consecutive_correct = 0
    feedback: tuple[str, tuple[int, int, int], float] | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            timestamp = int(time.time() * 1000)
            detection = tracker.detect(frame, timestamp)
            body_view.update(frame, timestamp, detection[0] if detection else None)
            _draw_hands(frame, detection, tracker.other)
            new_sign = recognizer.update(
                detection, tracker.handedness_score, timestamp, tracker.other, body_view.body
            )
            _draw_recognizer(frame, recognizer, timestamp)
            body_view.draw(frame)

            if recognizer.rejected:
                feedback = (_two_hands_message(recognizer.rejected), (0, 200, 255), now + 2.0)
            if target is not None and new_sign is not None:
                if new_sign == target and not body_view.framed:
                    consecutive_correct = 0
                    message = f"Sena correcta, pero {body_view.status.issue.lower()}"
                    feedback = (message, (0, 200, 255), now + 2.0)
                elif new_sign == target:
                    consecutive_correct += 1
                    message = f"CORRECTO: {_ascii(target)} | intento {consecutive_correct}/3"
                    feedback = (message, (80, 220, 120), now + 1.5)
                else:
                    consecutive_correct = 0
                    message = f"Se detecto {_ascii(new_sign)}, se esperaba {_ascii(target)}"
                    feedback = (message, (0, 200, 255), now + 1.5)

            if feedback and now < feedback[2]:
                _draw_text(frame, feedback[0], 70, feedback[1])
            elif target is not None:
                _draw_text(frame, f"Objetivo: {_ascii(target)} | Q sale", 70)
            else:
                _draw_text(
                    frame,
                    f"Estatica: sostenla {HOLD_SECONDS:g} s | Movimiento: hazla completa | Q sale",
                    70,
                )

            cv2.imshow("Mecabite - prueba de vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        tracker.close()
        body_view.close()


def _draw_prompt(frame: np.ndarray, sign: str, position: int, total: int) -> None:
    """Seña pedida, grande, abajo a la izquierda."""
    text = f"Haz: {_ascii(sign)}  ({position}/{total})"
    origin = (16, frame.shape[0] - 28)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 14)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)


def _practice_rows(
    target: str,
    forced: bool,
    recognizer: Recognizer,
    hand_frames: deque[tuple[int, np.ndarray, bool, Hand | None, str]],
    now_ms: int,
    hold_ms: int | None = None,
) -> list[tuple[int, np.ndarray, bool, Hand | None]]:
    """Cuadros a guardar (t_ms, puntos, mano izquierda, la otra mano): la seña completa si es
    con movimiento; si es estática, los del intervalo pedido en que el modelo vio la letra
    pedida (o todos, si se forzó)."""
    if recognizer.is_dynamic(target):
        return recognizer.moving_detector.frames()
    since = now_ms - (hold_ms if hold_ms is not None else HOLD_SECONDS * 1000 + 300)
    return [
        (t_ms, points, is_left, other)
        for t_ms, points, is_left, other, label in hand_frames
        if t_ms >= since and (forced or label == target)
    ]


def _glove_packets_in_window(
    packets: list[tuple[int, int, list[float]]], start_ms: int, end_ms: int
) -> list[tuple[int, int, list[float]]]:
    """Paquetes BLE capturados dentro del intervalo de la seña, inclusive."""
    return [packet for packet in packets if start_ms <= packet[0] <= end_ms]


def _draw_glove_hold_progress(frame: np.ndarray, sign: str, elapsed_ms: int) -> None:
    """Barra de progreso para confirmar una seña estática durante un segundo."""
    progress = min(max(elapsed_ms / GLOVE_HOLD_MS, 0.0), 1.0)
    elapsed = min(max(elapsed_ms, 0), GLOVE_HOLD_MS) / 1000
    _draw_text(frame, f"Mantén {_ascii(sign)}: {elapsed:.1f}/1.0 s", 108, (80, 220, 120))
    x, y = 16, 118
    width, height = max(1, min(440, frame.shape[1] - 32)), 18
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    cv2.rectangle(
        frame,
        (x, y),
        (x + round(width * progress), y + height),
        (80, 220, 120),
        -1,
    )
    cv2.rectangle(frame, (x, y), (x + width, y + height), (235, 235, 235), 1)
    if progress >= 1.0:
        _draw_text(frame, "Capturando seña y guante...", 160, (80, 220, 120))


def _clip_since(
    clip: deque[tuple[int, np.ndarray]], since_ms: float
) -> tuple[list[np.ndarray], float]:
    """Cuadros del clip desde `since_ms` y sus cuadros por segundo."""
    frames = [frame for t_ms, frame in clip if t_ms >= since_ms]
    if len(frames) < 2:
        return [], 0.0
    return frames, len(frames) / max((clip[-1][0] - since_ms) / 1000, 0.1)


def _practice_clip_duration_ms(target: str) -> int:
    """Ventana de video: 1 s para movimiento y 2 s para palabras del nivel 3."""
    sign = load_sign_catalog().find(target)
    return LEVEL_3_WORD_CLIP_MS if sign is not None and sign.level == 3 else MOVEMENT_CLIP_MS


def _sample_glove_packets(
    packets: list[tuple[int, int, list[float]]], interval_ms: int = GLOVE_REVIEW_SAMPLE_MS
) -> list[tuple[int, int, list[float]]]:
    """Muestra el paquete más cercano cada 200 ms; el CSV conserva el flujo completo."""
    ordered = sorted(packets, key=lambda packet: packet[0])
    if len(ordered) <= 1:
        return ordered
    samples = []
    next_index = 0
    target_ms = ordered[0][0]
    while target_ms < ordered[-1][0] and next_index < len(ordered):
        nearest_index = min(
            range(next_index, len(ordered)),
            key=lambda index: abs(ordered[index][0] - target_ms),
        )
        samples.append(ordered[nearest_index])
        next_index = nearest_index + 1
        target_ms += interval_ms
    return samples


def _glove_review_panel(
    packets: list[tuple[int, int, list[float]]], capture_height: int, page: int = 0
) -> np.ndarray:
    """Tabla compacta de 10 paquetes espaciados 200 ms; páginas para intervalos largos."""
    width = 800
    samples = _sample_glove_packets(packets)
    page_count = max(1, (len(samples) + GLOVE_REVIEW_PAGE_SIZE - 1) // GLOVE_REVIEW_PAGE_SIZE)
    page = min(max(page, 0), page_count - 1)
    start = page * GLOVE_REVIEW_PAGE_SIZE
    visible_samples = samples[start : start + GLOVE_REVIEW_PAGE_SIZE]
    first_row, line_height = 110, 24
    panel_height = max(capture_height, first_row + len(visible_samples) * line_height + 22)
    panel = np.full((panel_height, width, 3), (28, 31, 36), dtype=np.uint8)
    counts: Counter[int] = Counter(emitter for _, emitter, _ in packets)
    scale, header_scale = 0.4, 0.3

    def put(
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int] = (235, 235, 235),
        font_scale: float = scale,
    ) -> None:
        cv2.putText(
            panel,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, round(font_scale * 3)),
            cv2.LINE_AA,
        )

    put("DATOS BLE DE LA CAPTURA", 14, 30, (100, 220, 140))
    put(
        f"Paquetes: {len(packets)}   E1: {counts[1]}   E2: {counts[2]}   "
        f"Muestras: {start + 1}-{start + len(visible_samples)}/{len(samples)} "
        f"cada {GLOVE_REVIEW_SAMPLE_MS} ms  Pagina: {page + 1}/{page_count}",
        14,
        60,
    )
    header_y = 91
    time_x, emitter_x, ok_x, values_x, value_gap = 10, 104, 126, 150, 57
    put("timestamp_ms", time_x, header_y, (170, 185, 205), header_scale)
    put("E", emitter_x, header_y, (170, 185, 205), header_scale)
    put("OK", ok_x, header_y, (170, 185, 205), header_scale)
    for index, sensor in enumerate(GLOVE_VALUES):
        put(sensor, values_x + index * value_gap, header_y, (170, 185, 205), header_scale)

    if not samples:
        put("No llegaron paquetes BLE en este intervalo.", 14, first_row + line_height)
    else:
        for row_index, (t_ms, emitter, values) in enumerate(visible_samples):
            y = first_row + row_index * line_height
            color = (120, 225, 155) if emitter == 1 else (120, 190, 245)
            put(str(t_ms), time_x, y, color)
            put(str(emitter), emitter_x, y, color)
            put("OK", ok_x, y, (100, 220, 140))
            for index, value in enumerate(values):
                put(f"{value:.1f}", values_x + index * value_gap, y, color)
        put(
            "OK = valido | , anterior / . siguiente pagina",
            14,
            panel_height - 8,
            (100, 220, 140),
        )
    return panel


def _review(
    frames: list[np.ndarray],
    fps: float,
    title: str,
    size: tuple[int, int],
    glove_packets: list[tuple[int, int, list[float]]] | None = None,
) -> str:
    """Muestra la captura antes de guardarla (el clip se repite en bucle).

    Devuelve `save`, `retry`, `skip` o `quit`.
    """
    delay = max(int(1000 / fps), 1) if len(frames) > 1 else 50
    index = 0
    glove_page = 0
    while True:
        frame = frames[index % len(frames)]
        if glove_packets is None:
            view = cv2.resize(frame, size)
            _draw_text(view, title, 32, (80, 220, 120))
            _draw_text(view, "ENTER/ESPACIO guarda | R la repite | N salta | Q sale", 70)
        else:
            panel_width = 800
            max_width, max_height = 1450, 760
            scale = min(
                1.0,
                max_height / frame.shape[0],
                (max_width - panel_width) / frame.shape[1],
            )
            capture_size = (
                max(1, round(frame.shape[1] * scale)),
                max(1, round(frame.shape[0] * scale)),
            )
            capture_view = cv2.resize(frame, capture_size)
            _draw_text(capture_view, title, 32, (80, 220, 120))
            _draw_text(capture_view, "ENTER/ESPACIO guarda | R la repite | N salta | Q sale", 70)
            panel = _glove_review_panel(glove_packets, capture_view.shape[0], glove_page)
            view_height = max(capture_view.shape[0], panel.shape[0])
            view = np.zeros(
                (view_height, capture_view.shape[1] + panel_width, 3), dtype=np.uint8
            )
            capture_y = (view_height - capture_view.shape[0]) // 2
            view[
                capture_y : capture_y + capture_view.shape[0], : capture_view.shape[1]
            ] = capture_view
            view[: panel.shape[0], capture_view.shape[1] :] = panel
        cv2.imshow(PRACTICE_WINDOW, view)
        key = cv2.waitKey(delay) & 0xFF
        if glove_packets is not None and key in (ord(","), 81):
            glove_page = max(0, glove_page - 1)
            continue
        if glove_packets is not None and key in (ord("."), 83):
            sample_count = len(_sample_glove_packets(glove_packets))
            page_count = max(
                1,
                (sample_count + GLOVE_REVIEW_PAGE_SIZE - 1) // GLOVE_REVIEW_PAGE_SIZE,
            )
            glove_page = min(page_count - 1, glove_page + 1)
            continue
        if key in (13, 10, ord(" ")):
            return "save"
        if key in (ord("r"), ord("R"), 8, 127):
            return "retry"
        if key in (ord("n"), ord("N")):
            return "skip"
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        index += 1


def _capture_meta(
    target: str,
    participant: str,
    result: str,
    image: np.ndarray,
    recognizer: Recognizer,
    status: BodyStatus | None,
) -> dict[str, object]:
    """Datos del momento de la captura, para revisar o limpiar el dataset después."""
    prediction = None
    if recognizer.is_dynamic(target) and recognizer.moving_detector.last:
        prediction = recognizer.moving_detector.last
    elif recognizer.frame_label:
        prediction = recognizer.frame_label
    models = {
        path.stem: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))
        for path in (STATIC_MODEL_PATH, DYNAMIC_MODEL_PATH)
        if path.exists()
    }
    return {
        "sena": target,
        "persona": participant,
        "resultado": result,
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resolucion": [image.shape[1], image.shape[0]],
        "espejo": True,
        "prediccion": (
            {"sena": prediction[0], "confianza": round(prediction[1], 3)} if prediction else None
        ),
        "encuadre": "ok" if status is None or status.framed else status.issue,
        "mano_en": status.location if status else None,
        "modelos": models,
    }


def _save_practice(
    target: str,
    participant: str,
    rows: list[tuple[int, np.ndarray, bool, Hand | None]],
    image: np.ndarray,
    clip_frames: list[np.ndarray],
    fps: float,
    bodies: dict[int, np.ndarray] | None = None,
    meta: dict[str, object] | None = None,
    glove_packets: list[tuple[int, int, list[float]]] | None = None,
) -> str:
    """Guarda los puntos (para entrenar), una captura limpia (foto si es estática, clip si se
    mueve) y un .json con los datos del momento. Devuelve la ruta de la captura."""
    bodies = bodies or {}
    name = f"{participant}{PRACTICE_TAG}{int(time.time() * 1000)}"
    landmarks_path = PRACTICE_DIR / "landmarks" / target / f"{name}.csv"
    landmarks_path.parent.mkdir(parents=True, exist_ok=True)
    with landmarks_path.open("w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(RECORD_HEADER)
        for t_ms, points, is_left, other in rows:
            writer.writerow(_record_row(t_ms, points, is_left, bodies.get(t_ms), other))

    glove_path = None
    if glove_packets is not None:
        glove_path = PRACTICE_DIR / "glove" / participant / target / name / "glove.csv"
        write_packets(glove_path, glove_packets)

    capture_dir = PRACTICE_DIR / "captures" / target
    capture_dir.mkdir(parents=True, exist_ok=True)
    if clip_frames:
        height, width = clip_frames[0].shape[:2]
        path = capture_dir / f"{name}.mp4"
        video = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height)
        )
        if not video.isOpened():
            video.release()
            raise RuntimeError(f"No se pudo crear el video de práctica: {path}")
        for frame in clip_frames:
            video.write(np.ascontiguousarray(frame, dtype=np.uint8))
        video.release()
    else:
        path = capture_dir / f"{name}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    relative = path.relative_to(ROOT_DIR).as_posix()
    sidecar = {
        **(meta or {}),
        "captura": relative,
        "puntos": landmarks_path.relative_to(ROOT_DIR).as_posix(),
    }
    if glove_path is not None:
        sidecar["guante"] = glove_path.relative_to(ROOT_DIR).as_posix()
    path.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return relative


def _log_practice(participant: str, sign: str, result: str, seconds: float, capture: str) -> None:
    log_path = PRACTICE_DIR / PRACTICE_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    with log_path.open("a", newline="") as file_handle:
        writer = csv.writer(file_handle)
        if new_file:
            writer.writerow(["fecha", "persona", "sena", "resultado", "segundos", "captura"])
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([stamp, participant, sign, result, f"{seconds:.1f}", capture])


def practice(
    participant: str,
    camera: int,
    signs: list[str] | None,
    reps: int,
    review: bool = True,
    body_enabled: bool = True,
    glove_enabled: bool = False,
) -> None:
    """Pide seña por seña. Cuando el modelo reconoce la pedida, muestra la captura para
    aprobarla (con `review`), guarda sus puntos y la captura en data/practice/ y pasa a la
    siguiente. `vision train` usa lo guardado como datos de visión. Con `glove_enabled`, también
    guarda `glove.csv` crudo por repetición para prepararlo con el pipeline de `manitas`.

    Con `body_enabled`, solo se guarda si la persona está bien encuadrada (de frente, con
    cara y hombros a la vista), y se guardan también los puntos del cuerpo."""
    recognizer = _load_recognizer()
    known = recognizer.signs
    sign_catalog = load_sign_catalog()
    requested = _resolve_signs(signs, known) if signs else known
    queue = [sign for sign in requested for _ in range(reps)]
    print(f"Práctica de {participant}: {' '.join(queue)}")

    tracker, capture = HandTracker(), open_camera(camera)
    body_view = BodyView(body_enabled)
    glove = (
        GloveCollector(settings.glove_device_name, settings.glove_characteristic_uuid)
        if glove_enabled
        else None
    )
    if glove:
        glove.start()
    hand_frames: deque[tuple[int, np.ndarray, bool, Hand | None, str]] = deque(maxlen=120)
    body_frames: deque[tuple[int, np.ndarray]] = deque(maxlen=120)
    clip: deque[tuple[int, np.ndarray]] = deque()
    results: list[tuple[str, str]] = []
    feedback: tuple[str, tuple[int, int, int], float] | None = None
    index, started = 0, time.monotonic()
    glove_mark = glove.mark() if glove else 0
    glove_hold_started_ms: int | None = None
    paused_until = 0.0  # tras guardar, un momento para cambiar de seña (o reacomodar la mano)

    try:
        while index < len(queue):
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            clean = frame.copy()
            now = time.monotonic()
            timestamp = int(time.time() * 1000)
            target = queue[index]
            dynamic = recognizer.is_dynamic(target)
            sign_info = sign_catalog.find(target) if dynamic else None
            dynamic_timeout_ms = (
                LEVEL_3_DETECTION_TIMEOUT_MS
                if sign_info is not None and sign_info.level == 3
                else None
            )
            if dynamic:
                clip_height = int(clean.shape[0] * CLIP_WIDTH / clean.shape[1])
                clip_height = max(2, clip_height - clip_height % 2)
                clip.append((timestamp, cv2.resize(clean, (CLIP_WIDTH, clip_height))))
                while timestamp - clip[0][0] > CLIP_HISTORY_MS:
                    clip.popleft()
            if now < paused_until:
                _draw_prompt(frame, target, index + 1, len(queue))
                if feedback:
                    _draw_text(frame, feedback[0], 70, feedback[1])
                cv2.imshow(PRACTICE_WINDOW, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
                started = now
                continue

            detection = tracker.detect(frame, timestamp)
            body_view.update(frame, timestamp, detection[0] if detection else None)
            if body_view.body is not None:
                body_frames.append((timestamp, body_view.body))
            _draw_hands(frame, detection, tracker.other)
            new_sign = recognizer.update(
                detection,
                tracker.handedness_score,
                timestamp,
                tracker.other,
                body_view.body,
                max_movement_duration_ms=dynamic_timeout_ms,
            )
            hold_elapsed_ms: int | None = None
            auto_capture = False
            if glove_enabled and not dynamic:
                if recognizer.stabilizer.candidate == target:
                    if glove_hold_started_ms is None:
                        glove_hold_started_ms = round(recognizer.stabilizer.since * 1000)
                    hold_elapsed_ms = max(0, timestamp - glove_hold_started_ms)
                    auto_capture = hold_elapsed_ms >= GLOVE_HOLD_MS
                else:
                    glove_hold_started_ms = None
            else:
                glove_hold_started_ms = None
            if detection and recognizer.frame_label:
                label = recognizer.frame_label[0]
                hand_frames.append((timestamp, *detection, tracker.other, label))
            _draw_recognizer(frame, recognizer, timestamp)
            body_view.draw(frame)
            _draw_prompt(frame, target, index + 1, len(queue))
            if recognizer.rejected:
                feedback = (_two_hands_message(recognizer.rejected), (0, 200, 255), now + 2.0)
            if new_sign is not None and new_sign != target:
                message = f"Detecte {_ascii(new_sign)}, se pide {_ascii(target)}"
                feedback = (message, (0, 200, 255), now + 1.5)
            if feedback and now < feedback[2]:
                _draw_text(frame, feedback[0], 70, feedback[1])
            else:
                if dynamic:
                    how = "hazla completa y deja la mano quieta"
                elif glove_enabled:
                    how = "sostenla 1 segundo para capturar"
                else:
                    how = "sostenla"
                _draw_text(frame, f"{how} | ESPACIO guarda igual | N salta | Q sale", 70)
            if hold_elapsed_ms is not None:
                _draw_glove_hold_progress(frame, target, hold_elapsed_ms)
            cv2.imshow(PRACTICE_WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            result = None
            if auto_capture or (new_sign == target and not (glove_enabled and not dynamic)):
                result = "reconocida"
            elif key == ord(" "):
                result = "forzada"  # el modelo no la reconoció, pero tú sabes que está bien
            elif key in (ord("n"), ord("N")):
                result = "saltada"
            if result is None:
                continue

            elapsed = now - started  # hasta reconocerla, sin contar la revisión
            saved = ""
            if result != "saltada" and not body_view.framed:
                # La seña debe hacerse frente a la cámara: si no, no se guarda (RF-03).
                feedback = (f"No se guardo: {body_view.status.issue}", (0, 200, 255), now + 2.0)
                recognizer.reset()
                hand_frames.clear()
                glove_hold_started_ms = None
                continue
            if result != "saltada":
                rows = _practice_rows(
                    target,
                    result == "forzada",
                    recognizer,
                    hand_frames,
                    timestamp,
                    hold_ms=GLOVE_HOLD_MS if glove_enabled and not dynamic else None,
                )
                if not rows:
                    feedback = ("No hay mano: no se guardo nada", (0, 200, 255), now + 1.5)
                    glove_hold_started_ms = None
                    continue
                capture_duration_ms = (
                    _practice_clip_duration_ms(target) if dynamic else GLOVE_CAPTURE_WINDOW_MS
                )
                capture_start_ms = timestamp - capture_duration_ms
                if dynamic:
                    movement_rows = _trim_to_motion(rows)
                    if movement_rows:
                        capture_start_ms = min(capture_start_ms, movement_rows[0][0])
                glove_window_start_ms = capture_start_ms
                glove_packets = (
                    _glove_packets_in_window(
                        glove.since(glove_mark), glove_window_start_ms, timestamp
                    )
                    if glove
                    else None
                )
                if glove and len(glove_packets) < MIN_GLOVE_PACKETS:
                    feedback = (
                        f"No hay suficientes datos del guante ({len(glove_packets)} paquetes)",
                        (0, 200, 255),
                        now + 2.0,
                    )
                    recognizer.reset()
                    hand_frames.clear()
                    glove_hold_started_ms = None
                    continue
                if dynamic:
                    clip_frames, fps = _clip_since(clip, capture_start_ms)
                else:
                    clip_frames, fps = [], 0.0
                if review:
                    photo = clean.copy()
                    if body_view.body is not None:
                        draw_body(photo, body_view.body)
                    _draw_hands(photo, rows[-1][1:3], rows[-1][3])
                    size = (clean.shape[1], clean.shape[0])
                    title = f"{_ascii(target)} ({result}): revisala antes de guardar"
                    decision = _review(
                        clip_frames or [photo],
                        fps,
                        title,
                        size,
                        glove_packets=glove_packets,
                    )
                    if decision == "quit":
                        break
                    if decision == "retry":
                        # Lo descartado también queda en el registro: sirve para ver errores.
                        _log_practice(participant, target, "descartada", elapsed, "")
                        recognizer.reset()
                        hand_frames.clear()
                        clip.clear()
                        glove_hold_started_ms = None
                        feedback = (
                            "Descartada: hazla otra vez",
                            (0, 200, 255),
                            time.monotonic() + 1.5,
                        )
                        continue
                    if decision == "skip":
                        result = "saltada"
                if result != "saltada":
                    meta = _capture_meta(
                        target, participant, result, clean, recognizer, body_view.status
                    )
                    saved = _save_practice(
                        target,
                        participant,
                        rows,
                        clean,
                        clip_frames,
                        fps,
                        dict(body_frames),
                        meta,
                        glove_packets,
                    )
                    message = f"Guardada {_ascii(target)} ({result})"
                    feedback = (message, (80, 220, 120), time.monotonic() + SAVED_MESSAGE_SECONDS)
            now = time.monotonic()
            _log_practice(participant, target, result, elapsed, saved)
            results.append((target, result))
            index, started = index + 1, now
            glove_mark = glove.mark() if glove else 0
            glove_hold_started_ms = None
            if saved:
                paused_until = now + SAVED_MESSAGE_SECONDS
            recognizer.reset()
            hand_frames.clear()
            clip.clear()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        tracker.close()
        body_view.close()
        if glove:
            glove.close()

    if not results:
        return
    summary = Counter(result for _, result in results)
    print("Resumen: " + ", ".join(f"{result}={count}" for result, count in summary.items()))
    hard = sorted({sign for sign, result in results if result != "reconocida"}, key=alphabet_key)
    if hard:
        print(f"No reconocidas a la primera: {' '.join(hard)}")
    if summary["reconocida"] + summary["forzada"]:
        print("Para que el modelo aprenda de lo guardado: vision train")


def clear_practice(yes: bool) -> None:
    """Borra todo lo guardado por `vision practice` (nunca los datasets de data/vision)."""
    files = [path for path in PRACTICE_DIR.rglob("*") if path.is_file()]
    if not files:
        print("No hay datos de práctica.")
        return
    landmarks = sum(
        path.suffix == ".csv" and path.parent.parent.name == "landmarks" for path in files
    )
    captures = sum(path.suffix in (".jpg", ".mp4") for path in files)
    print(
        f"Se borrará {PRACTICE_DIR.relative_to(ROOT_DIR)}: {landmarks} grabaciones de puntos, "
        f"{captures} capturas (con sus .json) y el registro de sesiones."
    )
    if not yes:
        answer = input("¿Borrar? [s/N] ").strip().lower()
        if answer not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado.")
            return
    shutil.rmtree(PRACTICE_DIR)
    print("Borrado. El modelo actual aún los incluye hasta que corras: vision train")


def extract_image(image_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"No se pudo leer la imagen: {image_path}")
    tracker = HandTracker(images=True, mirrored=False)
    try:
        detection = tracker.detect(image)
    finally:
        tracker.close()
    if detection is None:
        print('{"detected": false}')
        return
    points, is_left = detection
    print(json.dumps({"detected": True, "is_left": is_left, "landmarks": points.round(6).tolist()}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="recolectar landmarks desde la webcam")
    collect_parser.add_argument("--label", required=True, help="clase, por ejemplo A o reposo")
    collect_parser.add_argument("--participant", default="p01")
    collect_parser.add_argument("--camera", type=int, default=0)
    collect_parser.add_argument(
        "--no-body", action="store_true", help="sin puntos del cuerpo ni revisión de encuadre"
    )

    train_parser = subparsers.add_parser("train", help="entrenar los modelos estático y dinámico")
    train_parser.add_argument("--model", choices=BACKENDS, default="mlp")
    train_parser.add_argument("--only", choices=("static", "dynamic"))

    live_parser = subparsers.add_parser("live", help="probar el modelo desde la webcam")
    live_parser.add_argument("--target", default="A")
    live_parser.add_argument("--camera", type=int, default=0)
    live_parser.add_argument(
        "--no-body", action="store_true", help="sin puntos del cuerpo ni revisión de encuadre"
    )

    demo_parser = subparsers.add_parser("demo", help="ver en vivo qué seña reconoce el modelo")
    demo_parser.add_argument("--camera", type=int, default=0)
    demo_parser.add_argument(
        "--no-body", action="store_true", help="sin puntos del cuerpo ni revisión de encuadre"
    )

    practice_parser = subparsers.add_parser(
        "practice", help="pide seña por seña y guarda cada una reconocida para reentrenar"
    )
    practice_parser.add_argument("--participant", default="p01")
    practice_parser.add_argument(
        "--signs", help="señas a pedir separadas por coma (ej. A,B,Ñ); por defecto todas"
    )
    practice_parser.add_argument("--reps", type=int, default=1, help="veces seguidas por seña")
    practice_parser.add_argument("--camera", type=int, default=0)
    practice_parser.add_argument(
        "--no-body", action="store_true", help="sin puntos del cuerpo ni revisión de encuadre"
    )
    practice_parser.add_argument(
        "--no-review", action="store_true", help="guardar sin mostrar la captura antes"
    )

    glove_practice_parser = subparsers.add_parser(
        "practica_guante",
        aliases=["practice-glove", "practice_glove"],
        help="practicar y guardar landmarks de visión más paquetes crudos del guante",
    )
    glove_practice_parser.add_argument("--participant", default="p01")
    glove_practice_parser.add_argument(
        "--signs", help="señas a pedir separadas por coma (ej. A,B,Ñ); por defecto todas"
    )
    glove_practice_parser.add_argument("--reps", type=int, default=1)
    glove_practice_parser.add_argument("--camera", type=int, default=0)
    glove_practice_parser.add_argument("--no-body", action="store_true")
    glove_practice_parser.add_argument("--no-review", action="store_true")

    reference_parser = subparsers.add_parser(
        "glove-reference",
        help="inclinación del guante por seña para las correcciones (de grabaciones o en vivo)",
    )
    reference_parser.add_argument("--sign", help="solo esta seña (obligatoria con --live)")
    reference_parser.add_argument(
        "--live", action="store_true", help="medir sosteniendo la seña con el guante conectado"
    )
    reference_parser.add_argument("--seconds", type=float, default=3.0)
    reference_parser.add_argument(
        "--source", type=Path, default=PRACTICE_DIR / "glove", help="grabaciones de practica_guante"
    )

    clear_parser = subparsers.add_parser(
        "clear", help="borrar todo lo guardado por practice (no toca los datasets)"
    )
    clear_parser.add_argument("-y", "--yes", action="store_true", help="no pedir confirmación")

    image_parser = subparsers.add_parser("extract", help="extraer landmarks de una imagen")
    image_parser.add_argument("--image", type=Path, required=True)

    videos_parser = subparsers.add_parser(
        "import-videos", help="extraer landmarks de un dataset de videos (ej. MSL-dynamic-signs)"
    )
    videos_parser.add_argument(
        "--source", type=Path, default=RAW_DIR / "MSL-dynamic-signs", help="carpeta con .mp4"
    )
    videos_parser.add_argument("--stride", type=int, default=1, help="procesar 1 de cada N cuadros")
    videos_parser.add_argument("--label", help="etiqueta si los nombres no usan formato MSL")
    videos_parser.add_argument(
        "--participant", default="p01", help="persona para videos con etiqueta manual"
    )
    videos_parser.add_argument("--overwrite", action="store_true")

    glosses_parser = subparsers.add_parser(
        "import-glosses",
        help="importar palabras del dataset de glosas de Zenodo (mano y cuerpo)",
    )
    glosses_parser.add_argument("--source", type=Path, default=RAW_DIR / "MSL-glosses")
    glosses_parser.add_argument("--overwrite", action="store_true")

    frames_parser = subparsers.add_parser(
        "import-frames", help="importar secuencias de fotos (una subcarpeta por persona)"
    )
    frames_parser.add_argument("--source", type=Path, required=True)
    frames_parser.add_argument("--label", required=True, help="carpeta de la seña, ej. mama")
    frames_parser.add_argument("--fps", type=float, default=8.0, help="ritmo supuesto de las fotos")
    frames_parser.add_argument("--prefix", default="M", help="prefijo de persona del dataset")
    frames_parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "collect":
        collect(args.label, args.participant, args.camera, not args.no_body)
    elif args.command == "import-videos":
        import_videos(
            args.source, max(args.stride, 1), args.overwrite, args.label, args.participant
        )
    elif args.command == "import-glosses":
        import_glosses(args.source, args.overwrite)
    elif args.command == "import-frames":
        import_frames(args.source, args.label, args.fps, args.prefix, args.overwrite)
    elif args.command == "train":
        try:
            train(args.model, args.only)
        except KeyboardInterrupt:
            raise SystemExit(
                "\nEntrenamiento cancelado. Los modelos que no alcanzaron a guardarse siguen "
                "como estaban."
            ) from None
    elif args.command == "live":
        live(args.target, args.camera, not args.no_body)
    elif args.command == "demo":
        live(None, args.camera, not args.no_body)
    elif args.command == "practice":
        signs = args.signs.split(",") if args.signs else None
        practice(
            args.participant,
            args.camera,
            signs,
            max(args.reps, 1),
            review=not args.no_review,
            body_enabled=not args.no_body,
        )
    elif args.command in ("practica_guante", "practice-glove", "practice_glove"):
        signs = args.signs.split(",") if args.signs else None
        practice(
            args.participant,
            args.camera,
            signs,
            max(args.reps, 1),
            review=not args.no_review,
            body_enabled=not args.no_body,
            glove_enabled=True,
        )
    elif args.command == "glove-reference":
        from app.vision.glove_reference import run

        run(args.sign, args.live, args.seconds, args.source)
    elif args.command == "clear":
        clear_practice(args.yes)
    else:
        extract_image(args.image)


if __name__ == "__main__":
    main()
