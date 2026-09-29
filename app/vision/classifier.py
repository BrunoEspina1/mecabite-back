"""Clasificadores ligeros para el primer prototipo de visión."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BACKENDS = ("rf", "et", "svm")


def make_backend(name: str):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42)
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

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"backend": self.backend, "labels": self.labels}, path)

    @classmethod
    def load(cls, path: Path) -> SignClassifier:
        payload = joblib.load(path)
        classifier = cls(payload["backend"])
        classifier.labels = payload["labels"]
        return classifier
