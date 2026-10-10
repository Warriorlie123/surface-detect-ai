"""Annotator boxes for the bundled NEU-DET samples, and a simple comparison
between those boxes and the model's predictions.

This is meant for looking at one image at a time in the app, not for
computing mAP (use scripts/evaluate.py for that).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .detection import Detection


def load_labels(path: Path) -> dict[str, list[Detection]]:
    """labels.json written by scripts/build_sample_library.py -> {file name: boxes}.
    Ground-truth boxes are stored as Detections with conf=1.0."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {name: [Detection(*b["box"], label=b["label"], conf=1.0) for b in boxes] for name, boxes in raw.items()}


def iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Comparison:
    found: int  # annotated boxes matched by a prediction of the same class
    missed: list[Detection]  # annotated boxes with no matching prediction
    extra: list[Detection]  # predictions that match no annotated box
    total: int  # annotated boxes


def compare(preds: list[Detection], truth: list[Detection], min_iou: float = 0.5) -> Comparison:
    """Greedy one-to-one matching, highest-confidence prediction first, same
    class and IoU >= min_iou (the same rule mAP50 uses)."""
    unmatched = list(range(len(truth)))
    extra = []
    for p in sorted(preds, key=lambda d: -d.conf):
        best, best_iou = None, min_iou
        for i in unmatched:
            if truth[i].label != p.label:
                continue
            v = iou(p, truth[i])
            if v >= best_iou:
                best, best_iou = i, v
        if best is None:
            extra.append(p)
        else:
            unmatched.remove(best)
    return Comparison(
        found=len(truth) - len(unmatched),
        missed=[truth[i] for i in unmatched],
        extra=extra,
        total=len(truth),
    )
