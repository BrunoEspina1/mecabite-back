"""Clasificadores ligeros para el primer prototipo de visión."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BACKENDS = ("mlp", "rf", "et", "svm")


def make_backend(name: str):
    if name == "mlp":
        # Misma exactitud que RF con personas nuevas, pero pesa KB en vez de cientos de MB
        # y son dos capas densas: se puede portar tal cual al teléfono.
        return make_pipeline(
            StandardScaler(),
            MLPClassifier((128, 64), early_stopping=True, max_iter=300, random_state=42),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
        )
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=42))
    raise ValueError(f"Modelo desconocido: {name}. Opciones: {', '.join(BACKENDS)}")


class SignClassifier:
    def __init__(self, backend):
        self.backend = backend
        self.labels: list[str] = []

    def train(self, features: np.ndarray, labels: list[str]) -> None:
        self.labels = sorted(set(labels))
        encoded = np.array([self.labels.index(label) for label in labels])
        self.backend.fit(features, encoded)

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        probabilities = self.backend.predict_proba(feature_vector.reshape(1, -1))[0]
        index = int(np.argmax(probabilities))
        return self.labels[index], float(probabilities[index])

    def predict_batch(self, features: np.ndarray) -> list[str]:
        return [self.labels[index] for index in self.backend.predict(features)]

    def save(self, path: Path) -> None:
        """Se escribe a un archivo aparte y se reemplaza al final: si el guardado se corta
        (Ctrl+C), el modelo anterior queda intacto en lugar de un archivo a medias."""
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        try:
            joblib.dump({"backend": self.backend, "labels": self.labels}, partial)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> SignClassifier:
        payload = joblib.load(path)
        backend = payload["backend"]
        if "n_jobs" in backend.get_params():
            # En vivo se predice un cuadro a la vez; lanzar hilos por cuadro lo haría más lento.
            backend.set_params(n_jobs=1)
        classifier = cls(backend)
        classifier.labels = payload["labels"]
        return classifier
