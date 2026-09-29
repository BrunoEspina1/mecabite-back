"""Recolección, entrenamiento y prueba en vivo del clasificador de visión.

Ejemplos:
    python -m app.vision collect --label A --participant p01
    python -m app.vision collect --label reposo --participant p01
    python -m app.vision train
    python -m app.vision live --target A

Solo se guardan landmarks y timestamps; no se guarda video.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

from app.vision.classifier import BACKENDS, SignClassifier, make_backend
from app.vision.features import NUM_LANDMARKS, landmarks_to_vector
from app.vision.tracker import HandTracker, draw_hand, open_camera

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "vision"
MODEL_PATH = ROOT_DIR / "models" / "vision" / "teacher.joblib"
HEADER = ["t_ms", "is_left"] + [
    f"{axis}{index}" for index in range(NUM_LANDMARKS) for axis in "xyz"
]
CONFIDENCE_THRESHOLD = 0.65
VOTE_FRAMES = 12


def _draw_text(frame: np.ndarray, text: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def collect(label: str, participant: str, camera: int) -> None:
    output_dir = DATA_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker, capture = HandTracker(), open_camera(camera)
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

            if detection:
                points, is_left = detection
                draw_hand(frame, points)
                if writer:
                    writer.writerow([timestamp, int(is_left), *np.round(points.flatten(), 6)])
                    frames += 1

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
                    writer.writerow(HEADER)
                    frames = 0
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if file_handle:
            file_handle.close()
        capture.release()
        cv2.destroyAllWindows()
        tracker.close()


def load_dataset() -> tuple[np.ndarray, list[str], list[str]]:
    features, labels, groups = [], [], []
    for label_dir in sorted(DATA_DIR.glob("*/")):
        for path in sorted(label_dir.glob("*.csv")):
            with path.open(newline="") as file_handle:
                reader = csv.reader(file_handle)
                next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    points = np.asarray(row[2:], dtype=float).reshape(NUM_LANDMARKS, 3)
                    features.append(landmarks_to_vector(points, is_left=row[1] == "1"))
                    labels.append(label_dir.name)
                    groups.append(path.stem)
    return np.asarray(features), labels, groups


def train(model_name: str) -> None:
    features, labels, groups = load_dataset()
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise SystemExit("Se necesitan al menos dos clases, por ejemplo A y reposo.")
    print(f"Dataset: {len(features)} cuadros, clases: {', '.join(unique_labels)}")

    classifier = SignClassifier(make_backend(model_name))
    if len(set(groups)) >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(features, labels, groups))
        classifier.train(features[train_idx], [labels[index] for index in train_idx])
        predictions = [classifier.predict(features[index])[0] for index in test_idx]
        print(
            classification_report(
                [labels[index] for index in test_idx], predictions, zero_division=0
            )
        )
        print(f"Grupos de prueba: {sorted({groups[index] for index in test_idx})}")
    else:
        print("Aviso: aún hay pocas grabaciones para una evaluación separada por grupo.")

    classifier.train(features, labels)
    classifier.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")


def live(target: str, camera: int) -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No hay modelo. Ejecuta primero: python -m app.vision train")

    classifier = SignClassifier.load(MODEL_PATH)
    tracker, capture = HandTracker(), open_camera(camera)
    recent: deque[str | None] = deque(maxlen=VOTE_FRAMES)
    consecutive_correct = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            detection = tracker.detect(frame, int(time.time() * 1000))
            if detection:
                points, is_left = detection
                draw_hand(frame, points)
                label, confidence = classifier.predict(landmarks_to_vector(points, is_left))
                recent.append(label if confidence >= CONFIDENCE_THRESHOLD else None)
                _draw_text(frame, f"Prediccion: {label} ({confidence:.0%})", 32)
            else:
                recent.append(None)
                _draw_text(frame, "No se detecta una mano", 32, (0, 200, 255))

            votes = Counter(recent).most_common(1)
            accepted = bool(votes and votes[0][0] and votes[0][1] >= VOTE_FRAMES * 0.75)
            if accepted and votes[0][0] == target:
                consecutive_correct += 1
                _draw_text(
                    frame,
                    f"CORRECTO: {target} | intento {consecutive_correct}/3",
                    70,
                    (80, 220, 120),
                )
                recent.clear()
            elif accepted:
                consecutive_correct = 0
                _draw_text(
                    frame,
                    f"Se detecto {votes[0][0]}, se esperaba {target}",
                    70,
                    (0, 200, 255),
                )
                recent.clear()
            else:
                _draw_text(frame, f"Objetivo: {target} | mantén la seña", 70)

            cv2.imshow("Mecabite - prueba de vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        tracker.close()


def extract_image(image_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"No se pudo leer la imagen: {image_path}")
    tracker = HandTracker(images=True)
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

    train_parser = subparsers.add_parser("train", help="entrenar el clasificador")
    train_parser.add_argument("--model", choices=BACKENDS, default="rf")

    live_parser = subparsers.add_parser("live", help="probar el modelo desde la webcam")
    live_parser.add_argument("--target", default="A")
    live_parser.add_argument("--camera", type=int, default=0)

    image_parser = subparsers.add_parser("extract", help="extraer landmarks de una imagen")
    image_parser.add_argument("--image", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "collect":
        collect(args.label, args.participant, args.camera)
    elif args.command == "train":
        train(args.model)
    elif args.command == "live":
        live(args.target, args.camera)
    else:
        extract_image(args.image)


if __name__ == "__main__":
    main()
