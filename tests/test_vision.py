from collections import deque

import numpy as np

from app.vision import teacher
from app.vision.classifier import SignClassifier, make_backend
from app.vision.features import NUM_FEATURES, landmarks_to_vector


def test_landmarks_to_vector_has_stable_shape() -> None:
    landmarks = np.arange(63, dtype=float).reshape(21, 3)

    features = landmarks_to_vector(landmarks)

    assert features.shape == (NUM_FEATURES,)
    assert np.isfinite(features).all()


def test_load_static_dataset_adds_mirrored_copy(tmp_path, monkeypatch) -> None:
    label_dir = tmp_path / "A"
    label_dir.mkdir()
    points = np.random.default_rng(0).random(63).round(6)
    (label_dir / "msl-abc__S3.csv").write_text(
        ",".join(teacher.HEADER) + "\n" + ",".join(["0", "1", *map(str, points)]) + "\n"
    )
    monkeypatch.setattr(teacher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(teacher, "PRACTICE_DIR", tmp_path / "practice")

    features, labels, groups = teacher.load_static_dataset()

    assert features.shape == (2, NUM_FEATURES)
    assert not np.allclose(features[0], features[1])
    assert labels == ["A", "A"]
    assert groups == ["S3", "S3"]


def test_person_id_from_file_names() -> None:
    assert teacher._person("msl-abc__S3") == "S3"
    assert teacher._person("S12-Ñ-Perfil.3") == "S12"
    assert teacher._person("p01_1759000000000") == "p01"


def test_classifier_predicts_trained_labels() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, NUM_FEATURES))
    labels = ["A"] * 10 + ["reposo"] * 10
    classifier = SignClassifier(make_backend("rf"))
    classifier.train(features, labels)

    prediction, confidence = classifier.predict(features[0])

    assert prediction in {"A", "reposo"}
    assert 0 <= confidence <= 1


def test_practice_saves_landmarks_that_training_reads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(teacher, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(teacher, "DATA_DIR", tmp_path / "data" / "vision")
    monkeypatch.setattr(teacher, "PRACTICE_DIR", tmp_path / "data" / "practice")
    points = np.random.default_rng(0).random((21, 3))
    image = np.zeros((36, 64, 3), dtype=np.uint8)

    capture = teacher._save_practice("A", "omar", [(0, points, False)], image, [], 0.0)

    assert capture.startswith("data/practice/captures/A/omar__practica_")
    assert capture.endswith(".jpg")
    features, labels, groups = teacher.load_static_dataset(mirror=False)
    # Las grabaciones propias cuentan OWN_WEIGHT veces frente a los datasets.
    assert features.shape == (teacher.OWN_WEIGHT, NUM_FEATURES)
    assert labels == ["A"] * teacher.OWN_WEIGHT
    assert groups == ["omar"] * teacher.OWN_WEIGHT


def test_clear_only_removes_practice_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(teacher, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(teacher, "PRACTICE_DIR", tmp_path / "data" / "practice")
    dataset = tmp_path / "data" / "vision" / "A" / "msl-abc__S1.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("t_ms\n")
    teacher._save_practice(
        "A", "omar", [(0, np.zeros((21, 3)), False)], np.zeros((8, 8, 3), np.uint8), [], 0.0
    )

    teacher.clear_practice(yes=True)

    assert not (tmp_path / "data" / "practice").exists()
    assert dataset.exists()


def test_dynamic_practice_saves_clip_since_sign_start(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(teacher, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(teacher, "PRACTICE_DIR", tmp_path / "data" / "practice")
    clip = deque((t_ms, np.zeros((36, 64, 3), np.uint8)) for t_ms in range(0, 2000, 100))

    frames, fps = teacher._clip_since(clip, 1000)
    capture = teacher._save_practice(
        "J", "omar", [(1000, np.zeros((21, 3)), False)], frames[0], frames, fps
    )

    assert len(frames) == 10
    assert fps == 10 / 0.9
    assert capture.endswith(".mp4")
    assert (tmp_path / capture).stat().st_size > 0


def test_classifier_save_leaves_no_partial_file(tmp_path) -> None:
    rng = np.random.default_rng(0)
    classifier = SignClassifier(make_backend("rf"))
    classifier.train(rng.normal(size=(10, NUM_FEATURES)), ["A"] * 5 + ["B"] * 5)
    path = tmp_path / "static.joblib"

    classifier.save(path)

    assert SignClassifier.load(path).labels == ["A", "B"]
    assert list(tmp_path.iterdir()) == [path]
