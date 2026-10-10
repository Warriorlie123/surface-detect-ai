from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One predicted box in image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    conf: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def shifted(self, dx: float, dy: float) -> "Detection":
        return Detection(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy, self.label, self.conf)


def from_ultralytics(result) -> list[Detection]:
    names = result.names
    out = []
    for xyxy, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        out.append(Detection(*xyxy, label=names[int(cls)], conf=float(conf)))
    return out
