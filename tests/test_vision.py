import numpy as np

from app.vision.classifier import SignClassifier, make_backend
from app.vision.features import NUM_FEATURES, landmarks_to_vector


def test_landmarks_to_vector_has_stable_shape() -> None:
    landmarks = np.arange(63, dtype=float).reshape(21, 3)

    features = landmarks_to_vector(landmarks)

    assert features.shape == (NUM_FEATURES,)
    assert np.isfinite(features).all()


def test_classifier_predicts_trained_labels() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, NUM_FEATURES))
    labels = ["A"] * 10 + ["reposo"] * 10
    classifier = SignClassifier(make_backend("rf"))
    classifier.train(features, labels)

    prediction, confidence = classifier.predict(features[0])

    assert prediction in {"A", "reposo"}
    assert 0 <= confidence <= 1
