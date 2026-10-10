import pytest

from inspector import Calibration, Detection, InspectionPolicy, Verdict, decide

CAL = Calibration(200, 200, field_width_mm=50)  # 0.25 mm / px


def det(label="patches", conf=0.8, size=20):
    return Detection(10, 10, 10 + size, 10 + size, label, conf)


def policy(**kw):
    base = dict(reject_conf=0.5, review_conf=0.25, max_length_mm=10, max_coverage_pct=5, zero_tolerance={"inclusion"})
    base.update(kw)
    return InspectionPolicy(**base)


def test_nothing_found_passes():
    assert decide([], CAL, policy()).verdict is Verdict.PASS


def test_zero_tolerance_class_rejects_even_when_small():
    d = decide([det("inclusion", size=4)], CAL, policy())
    assert d.verdict is Verdict.REJECT
    assert "zero-tolerance" in d.reasons[0]


def test_low_confidence_zero_tolerance_goes_to_review():
    d = decide([det("inclusion", conf=0.3, size=4)], CAL, policy())
    assert d.verdict is Verdict.REVIEW


def test_long_defect_rejects():
    # 60 px * 0.25 = 15 mm > 10 mm, coverage 60*60/200^2 = 9% also > 5%
    d = decide([det(size=60)], CAL, policy(max_coverage_pct=100))
    assert d.verdict is Verdict.REJECT
    assert any("length" in r for r in d.reasons)


def test_coverage_rejects():
    d = decide([det(size=36)], CAL, policy(max_length_mm=100))  # 3.24 % coverage
    assert decide([det(size=36)], CAL, policy(max_length_mm=100, max_coverage_pct=3)).verdict is Verdict.REJECT
    assert d.verdict is Verdict.REVIEW


def test_small_confident_defect_is_review_not_reject():
    d = decide([det(size=8)], CAL, policy())  # 2 mm, 0.16 %
    assert d.verdict is Verdict.REVIEW
    assert len(d.confirmed) == 1


def test_below_review_threshold_is_ignored():
    assert decide([det(conf=0.1, size=100)], CAL, policy()).verdict is Verdict.PASS


def test_bad_thresholds_raise():
    with pytest.raises(ValueError):
        InspectionPolicy(reject_conf=0.3, review_conf=0.6)
