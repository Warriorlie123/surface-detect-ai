"""Turn pixel boxes into physical measurements.

The camera is assumed to look straight down at the strip, so a single mm/px
scale works for the whole frame. Real lines would use a checkerboard
calibration to also correct lens distortion, but for a fixed top-down camera
the linear scale is a reasonable first approximation.
"""

from dataclasses import dataclass

import numpy as np

from .detection import Detection


@dataclass(frozen=True)
class Calibration:
    image_width_px: int
    image_height_px: int
    field_width_mm: float  # real width of the surface visible in the frame

    @property
    def mm_per_px(self) -> float:
        return self.field_width_mm / self.image_width_px

    @property
    def image_area_mm2(self) -> float:
        return self.image_width_px * self.image_height_px * self.mm_per_px**2


@dataclass(frozen=True)
class Measurement:
    label: str
    conf: float
    length_mm: float
    width_mm: float
    area_mm2: float


def measure(det: Detection, cal: Calibration) -> Measurement:
    w = det.width * cal.mm_per_px
    h = det.height * cal.mm_per_px
    # Bounding-box based, so a diagonal scratch is reported by its longer box
    # side, not its true length. Good enough for pass/fail, not for metrology.
    return Measurement(
        label=det.label,
        conf=det.conf,
        length_mm=max(w, h),
        width_mm=min(w, h),
        area_mm2=w * h,
    )


def defect_coverage(dets: list[Detection], cal: Calibration) -> tuple[float, float]:
    """Return (defect area in mm^2, percent of the image covered).

    Boxes are rasterised into a mask so overlapping detections are counted
    once. Summing box areas (what the first version did) double counts
    overlaps and can go above 100% on images with many boxes.
    """
    w, h = cal.image_width_px, cal.image_height_px
    if not dets or w == 0 or h == 0:
        return 0.0, 0.0

    mask = np.zeros((h, w), dtype=bool)
    for d in dets:
        x1 = int(np.clip(np.round(d.x1), 0, w))
        y1 = int(np.clip(np.round(d.y1), 0, h))
        x2 = int(np.clip(np.round(d.x2), 0, w))
        y2 = int(np.clip(np.round(d.y2), 0, h))
        mask[y1:y2, x1:x2] = True

    px = int(mask.sum())
    return px * cal.mm_per_px**2, 100.0 * px / (w * h)
