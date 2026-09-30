from collections import deque
from pathlib import Path

import numpy as np
import pytest

from app.vision import teacher
from app.vision.classifier import SignClassifier, make_backend
from app.vision.features import NUM_FEATURES, landmarks_to_vector
from app.vision.glove import GLOVE_VALUES, write_packets
from app.vision.tracker import HandSlots


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

    capture = teacher._save_practice("A", "omar", [(0, points, False, None)], image, [], 0.0)

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
        "A", "omar", [(0, np.zeros((21, 3)), False, None)], np.zeros((8, 8, 3), np.uint8), [], 0.0
    )

    teacher.clear_practice(yes=True)

    assert not (tmp_path / "data" / "practice").exists()
    assert dataset.exists()


def test_dynamic_practice_saves_clip_since_sign_start(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(teacher, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(teacher, "PRACTICE_DIR", tmp_path / "data" / "practice")
    red_frame = np.full((36, 64, 3), (10, 30, 220), np.uint8)
    clip = deque((t_ms, red_frame.copy()) for t_ms in range(0, 2000, 100))

    frames, fps = teacher._clip_since(clip, 1000)
    capture = teacher._save_practice(
        "J", "omar", [(1000, np.zeros((21, 3)), False, None)], frames[0], frames, fps
    )

    assert len(frames) == 10
    assert fps == 10 / 0.9
    assert capture.endswith(".mp4")
    video_path = tmp_path / capture
    assert video_path.stat().st_size > 0
    video = teacher.cv2.VideoCapture(str(video_path))
    ok, decoded = video.read()
    video.release()
    assert ok
    means = decoded.mean(axis=(0, 1))
    assert means[2] > means[1] + 60  # el canal rojo no se guarda con dominante verde


def test_custom_video_import_uses_requested_sign_and_participant() -> None:
    identity = teacher._video_import_identity(
        Path("mama_01_normal.mp4"), label="mama", participant="mama01"
    )

    assert identity == ("mama", "mama01_mama_mama_01_normal")


@pytest.mark.parametrize(
    ("target", "duration_ms"),
    [("J", 2000), ("K", 2000), ("hola", 4000), ("por_favor", 4000)],
)
def test_practice_clip_duration_depends_on_sign_level(target: str, duration_ms: int) -> None:
    assert teacher._practice_clip_duration_ms(target) == duration_ms


@pytest.mark.parametrize("duration_ms", [2000, 4000])
def test_dynamic_review_clip_spans_the_requested_duration(duration_ms: int) -> None:
    clip = deque((t_ms, np.zeros((4, 4, 3), np.uint8)) for t_ms in range(0, 5001, 100))

    frames, fps = teacher._clip_since(clip, 5000 - duration_ms)

    assert len(frames) == duration_ms // 100 + 1
    assert len(frames) / fps == pytest.approx(duration_ms / 1000)


def test_classifier_save_leaves_no_partial_file(tmp_path) -> None:
    rng = np.random.default_rng(0)
    classifier = SignClassifier(make_backend("rf"))
    classifier.train(rng.normal(size=(10, NUM_FEATURES)), ["A"] * 5 + ["B"] * 5)
    path = tmp_path / "static.joblib"

    classifier.save(path)

    assert SignClassifier.load(path).labels == ["A", "B"]
    assert list(tmp_path.iterdir()) == [path]


def test_dataset_people_are_not_own_recordings() -> None:
    assert teacher._person("G05-hola-glosas") == "G05"
    assert teacher._person("M03-mama-058-mother") == "M03"
    assert not teacher._is_own("G05")
    assert not teacher._is_own("S12")
    assert teacher._is_own("omar")


def test_trim_to_motion_keeps_only_the_sign() -> None:
    still = np.zeros((21, 3))
    still[:, 1] = np.linspace(0, 0.2, 21)  # palma de 0.2 de alto (muñeca -> nudillo medio)
    rows = []
    for index in range(90):  # 3 s a 30 fps: quieta, se mueve de 1.0 a 1.5 s, quieta
        t_ms = int(index * 1000 / 30)
        shift = np.clip((t_ms - 1000) / 500, 0, 1) * 0.6
        rows.append((t_ms, still + [shift, 0, 0], False, None))

    trimmed = teacher._trim_to_motion(rows)

    assert 400 <= trimmed[0][0] <= 700  # medio segundo antes de empezar a moverse
    # Termina al quedarse quieta: 150 ms de retraso al medir + END_STILL_MS + 50 ms.
    assert 1800 <= trimmed[-1][0] <= 2000


def test_resolve_signs_ignores_case_and_spaces() -> None:
    known = ["A", "Ñ", "hola", "por_favor"]

    assert teacher._resolve_signs(["a", "ñ", "HOLA", "Por favor"], known) == [
        "A",
        "Ñ",
        "hola",
        "por_favor",
    ]
    with pytest.raises(SystemExit):
        teacher._resolve_signs(["mama"], known)


def at(x: float, y: float = 0.5) -> tuple[np.ndarray, bool]:
    """Mano con los 21 puntos en (x, y)."""
    return np.tile([x, y, 0.0], (21, 1)), False


def test_hands_keep_their_place_when_mediapipe_swaps_them() -> None:
    left, right = at(0.3), at(1.2)
    slots = HandSlots()

    assert slots.assign([left, right], 0) == [left, right]
    assert slots.assign([right, left], 33) == [left, right]
    # Sola, cada una vuelve a su lugar; la derecha no pudo saltar hasta la izquierda.
    assert slots.assign([right], 66) == [None, right]
    far_left = at(0.3, 0.2)
    assert slots.assign([far_left], 100) == [far_left, None]


def test_hand_lost_for_a_while_can_reappear_far_away() -> None:
    entering, raised = at(1.1, 0.95), at(0.4, 0.4)
    slots = HandSlots()
    slots.assign([entering], 0)
    slots.assign([], 33)  # se pierde unos cuadros mientras sube

    assert slots.assign([raised], 66) == [raised, None]


def test_other_hand_becomes_main_only_after_losing_the_main_one() -> None:
    main, other = at(0.3), at(1.2)
    slots = HandSlots(promote_after_ms=300)
    slots.assign([main, other], 0)

    # La principal se pierde un momento (mano borrosa): la otra no la reemplaza.
    assert slots.assign([other], 100) == [None, other]
    assert slots.assign([main, other], 200) == [main, other]
    # Se va de verdad: pasado el tiempo, la otra es la principal.
    assert slots.assign([other], 300) == [None, other]
    assert slots.assign([other], 700) == [other, None]


def test_hand_coverage_prefers_hand_that_occupies_more_area() -> None:
    def palm(x: float, y: float) -> tuple[np.ndarray, bool]:
        points = np.zeros((21, 3))
        points[:, :2] = [x, y]
        points[:, 0] += np.linspace(-0.08, 0.08, 21)
        points[:, 1] += np.linspace(-0.06, 0.06, 21)
        return points, False

    moving = [palm(x, 0.3 + x / 2) for x in np.linspace(0.4, 1.0, 20)]
    blurred = [hand if index % 3 == 0 else None for index, hand in enumerate(moving)]
    still = [palm(0.7 + 0.01 * (index % 2), 0.5) for index in range(40)] + [palm(1.5, 0.5)]

    assert teacher._hand_coverage(blurred) > 5 * teacher._hand_coverage(still)


def test_record_row_matches_header_with_and_without_other_hand() -> None:
    points = np.zeros((21, 3))

    assert len(teacher._record_row(0, points, False, None)) == len(teacher.RECORD_HEADER)
    row = teacher._record_row(0, points, False, np.zeros((33, 4)), (points, True))
    assert len(row) == len(teacher.RECORD_HEADER)
    assert row[len(teacher.HEADER) + len(teacher.BODY_HEADER)] == 1


def test_write_glove_packets_uses_manitas_format(tmp_path) -> None:
    path = tmp_path / "glove.csv"
    write_packets(path, [(123, 2, list(range(len(GLOVE_VALUES))))])

    lines = path.read_text().splitlines()

    assert lines[0] == "t_ms,emisor," + ",".join(GLOVE_VALUES)
    assert lines[1].startswith("123,2,0,1,2")


def test_practice_rows_can_select_the_one_second_capture_window() -> None:
    class StaticRecognizer:
        @staticmethod
        def is_dynamic(_target: str) -> bool:
            return False

    points = np.zeros((21, 3))
    hand_frames = deque(
        (t_ms, points, False, None, "A") for t_ms in (999, 1000, 1500, 2000)
    )

    rows = teacher._practice_rows(
        "A", False, StaticRecognizer(), hand_frames, now_ms=2000, hold_ms=1000
    )

    assert [row[0] for row in rows] == [1000, 1500, 2000]


def test_glove_packets_are_limited_to_the_one_second_capture_window() -> None:
    packets = [
        (999, 1, [0.0] * len(GLOVE_VALUES)),
        (1000, 1, [1.0] * len(GLOVE_VALUES)),
        (1500, 2, [2.0] * len(GLOVE_VALUES)),
        (2000, 1, [3.0] * len(GLOVE_VALUES)),
        (2001, 2, [4.0] * len(GLOVE_VALUES)),
    ]

    captured = teacher._glove_packets_in_window(packets, start_ms=1000, end_ms=2000)

    assert [packet[0] for packet in captured] == [1000, 1500, 2000]


def test_glove_review_panel_renders_multiple_packet_rows(monkeypatch) -> None:
    rendered_text = []
    monkeypatch.setattr(
        teacher.cv2,
        "putText",
        lambda _image, text, *_args, **_kwargs: rendered_text.append(text),
    )
    start_ms = 1_750_000_000_000
    packets = [
        (start_ms + index * 200, index % 2 + 1, [float(index)] * len(GLOVE_VALUES))
        for index in range(21)
    ]

    samples = teacher._sample_glove_packets(packets)
    panel = teacher._glove_review_panel(packets, capture_height=300, page=0)

    assert len(samples) == 20
    assert [samples[index + 1][0] - samples[index][0] for index in range(19)] == [200] * 19
    assert panel.shape == (372, 800, 3)
    assert (
        "Paquetes: 21   E1: 11   E2: 10   Muestras: 1-10/20 cada 200 ms  Pagina: 1/2"
        in rendered_text
    )
    assert all(str(samples[index][0]) in rendered_text for index in range(10))
    assert str(samples[10][0]) not in rendered_text
    assert rendered_text.count("OK") == 11  # encabezado y las diez filas

    rendered_text.clear()
    teacher._glove_review_panel(packets, capture_height=300, page=1)
    assert all(str(samples[index][0]) in rendered_text for index in range(10, 20))
