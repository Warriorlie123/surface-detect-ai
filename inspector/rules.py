"""Pass / review / reject decision for one inspected image.

The first version rejected a part as soon as the model drew any box. That is
too strict for a real line: a 0.4 mm patch on a hot-rolled strip is usually
cosmetic, and low-confidence detections are exactly the cases a human should
look at. So the decision now has three outcomes:

    REJECT  - a confident detection that breaks one of the rules
    REVIEW  - something was found, but it is either low confidence or within
              tolerance; send it to an operator
    PASS    - nothing found above the review threshold
"""

from dataclasses import dataclass, field
from enum import Enum

from .detection import Detection
from .measure import Calibration, defect_coverage, measure


class Verdict(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


@dataclass
class InspectionPolicy:
    reject_conf: float = 0.50  # at or above this a detection counts as confirmed
    review_conf: float = 0.25  # between review_conf and reject_conf -> human review
    max_length_mm: float = 10.0  # any confirmed defect longer than this rejects
    max_coverage_pct: float = 5.0  # total confirmed defect area over this rejects
    zero_tolerance: set[str] = field(default_factory=lambda: {"inclusion"})

    def __post_init__(self):
        if self.review_conf > self.reject_conf:
            raise ValueError("review_conf must be <= reject_conf")


@dataclass
class Decision:
    verdict: Verdict
    reasons: list[str]
    confirmed: list[Detection]
    uncertain: list[Detection]


def decide(dets: list[Detection], cal: Calibration, policy: InspectionPolicy) -> Decision:
    confirmed = [d for d in dets if d.conf >= policy.reject_conf]
    uncertain = [d for d in dets if policy.review_conf <= d.conf < policy.reject_conf]
    reasons = []

    for d in confirmed:
        if d.label in policy.zero_tolerance:
            reasons.append(f"{d.label} is zero-tolerance")
            break

    longest = max((measure(d, cal).length_mm for d in confirmed), default=0.0)
    if longest > policy.max_length_mm:
        reasons.append(f"defect length {longest:.1f} mm > {policy.max_length_mm:g} mm")

    _, coverage = defect_coverage(confirmed, cal)
    if coverage > policy.max_coverage_pct:
        reasons.append(f"defect coverage {coverage:.2f}% > {policy.max_coverage_pct:g}%")

    if reasons:
        return Decision(Verdict.REJECT, reasons, confirmed, uncertain)

    if confirmed or uncertain:
        if confirmed:
            reasons.append(f"{len(confirmed)} defect(s) within tolerance")
        if uncertain:
            reasons.append(f"{len(uncertain)} low-confidence detection(s)")
        return Decision(Verdict.REVIEW, reasons, confirmed, uncertain)

    return Decision(Verdict.PASS, [], [], [])
