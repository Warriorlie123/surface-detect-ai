import json

from inspector.detection import Detection
from inspector.groundtruth import compare, iou, load_labels


def det(x1, y1, x2, y2, label="scratches", conf=0.9):
    return Detection(x1, y1, x2, y2, label, conf)


def test_iou():
    assert iou(det(0, 0, 10, 10), det(0, 0, 10, 10)) == 1.0
    assert iou(det(0, 0, 10, 10), det(20, 20, 30, 30)) == 0.0
    assert abs(iou(det(0, 0, 10, 10), det(5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_compare_matches_same_class_only():
    truth = [det(0, 0, 100, 100, "patches", 1.0), det(120, 120, 180, 180, "patches", 1.0)]
    preds = [
        det(2, 2, 98, 98, "patches", 0.8),  # matches the first box
        det(120, 120, 180, 180, "inclusion", 0.7),  # right place, wrong class
        det(0, 150, 20, 200, "patches", 0.4),  # nothing there
    ]
    c = compare(preds, truth)
    assert c.total == 2 and c.found == 1
    assert c.missed == [truth[1]]
    assert len(c.extra) == 2


def test_compare_is_one_to_one():
    truth = [det(0, 0, 100, 100, conf=1.0)]
    preds = [det(0, 0, 100, 100, conf=0.9), det(1, 1, 99, 99, conf=0.6)]
    c = compare(preds, truth)
    assert c.found == 1 and len(c.extra) == 1


def test_load_labels(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"scratches_1.jpg": [{"label": "scratches", "box": [1, 2, 3, 4]}]}))
    labels = load_labels(p)
    assert labels["scratches_1.jpg"] == [Detection(1, 2, 3, 4, "scratches", 1.0)]
    assert load_labels(tmp_path / "missing.json") == {}
