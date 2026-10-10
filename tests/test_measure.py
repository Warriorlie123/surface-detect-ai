import pytest

from inspector import Calibration, Detection, defect_coverage, measure


def box(x1, y1, x2, y2, label="scratches", conf=0.9):
    return Detection(x1, y1, x2, y2, label, conf)


def test_scale():
    cal = Calibration(200, 100, field_width_mm=50)
    assert cal.mm_per_px == pytest.approx(0.25)
    assert cal.image_area_mm2 == pytest.approx(50 * 25)


def test_measure_uses_longer_side_as_length():
    cal = Calibration(200, 200, field_width_mm=200)  # 1 mm / px
    m = measure(box(10, 10, 20, 50), cal)
    assert m.length_mm == pytest.approx(40)
    assert m.width_mm == pytest.approx(10)
    assert m.area_mm2 == pytest.approx(400)


def test_coverage_counts_overlap_once():
    cal = Calibration(100, 100, field_width_mm=100)
    a = box(0, 0, 50, 50)
    area, pct = defect_coverage([a, a, a], cal)
    assert area == pytest.approx(2500)
    assert pct == pytest.approx(25)


def test_coverage_partial_overlap():
    cal = Calibration(100, 100, field_width_mm=100)
    area, _ = defect_coverage([box(0, 0, 20, 20), box(10, 10, 30, 30)], cal)
    assert area == pytest.approx(400 + 400 - 100)


def test_coverage_clips_boxes_outside_image():
    cal = Calibration(100, 100, field_width_mm=100)
    _, pct = defect_coverage([box(-20, -20, 150, 150)], cal)
    assert pct == pytest.approx(100)


def test_coverage_empty():
    assert defect_coverage([], Calibration(10, 10, 10)) == (0.0, 0.0)
