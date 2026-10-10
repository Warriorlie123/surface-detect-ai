"""Model loading and prediction, with optional tiling for large images.

NEU-DET images are 200x200 crops. A real line camera produces frames that are
thousands of pixels wide, and if you feed one of those straight into the model
it gets resized to 640 px and small defects disappear. Tiling cuts the frame
into overlapping crops roughly the size the model was trained on, runs them as
one batch, shifts the boxes back to frame coordinates and merges duplicates
from the overlap regions with class-wise NMS.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms

from .detection import Detection, from_ultralytics

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = {
    "yolov8s_aug": ROOT / "weights" / "yolov8s_aug.pt",
    "yolov8n_baseline": ROOT / "weights" / "yolov8n_baseline.pt",
}


@lru_cache(maxsize=4)
def load_model(name: str):
    from ultralytics import YOLO  # slow import, keep it out of module load

    return YOLO(str(WEIGHTS[name]))


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets along one axis so tiles cover [0, length) with overlap.

    The last tile is pushed back to end exactly at the border instead of
    being padded, so every tile is full size.
    """
    if length <= tile:
        return [0]
    stride = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def merge(dets: list[Detection], iou: float = 0.5) -> list[Detection]:
    if not dets:
        return []
    labels = sorted({d.label for d in dets})
    boxes = torch.tensor([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d.conf for d in dets], dtype=torch.float32)
    idxs = torch.tensor([labels.index(d.label) for d in dets])
    keep = batched_nms(boxes, scores, idxs, iou)
    return [dets[i] for i in keep.tolist()]


def predict(model, image: Image.Image, conf: float, tile: int | None = None, overlap: float = 0.2) -> list[Detection]:
    image = image.convert("RGB")  # PNGs with alpha or 16-bit grayscale otherwise break the model

    if tile is None or (image.width <= tile and image.height <= tile):
        return from_ultralytics(model.predict(image, conf=conf, verbose=False)[0])

    arr = np.asarray(image)
    crops, offsets = [], []
    for y in tile_origins(image.height, tile, overlap):
        for x in tile_origins(image.width, tile, overlap):
            crops.append(arr[y : y + tile, x : x + tile])
            offsets.append((x, y))

    # ultralytics expects BGR for numpy input
    results = model.predict([c[..., ::-1] for c in crops], conf=conf, verbose=False)
    dets = []
    for res, (dx, dy) in zip(results, offsets):
        dets.extend(d.shifted(dx, dy) for d in from_ultralytics(res))
    return merge(dets)
