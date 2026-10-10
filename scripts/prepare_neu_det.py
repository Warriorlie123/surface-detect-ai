"""Convert NEU-DET (Pascal VOC XML) into a YOLO dataset.

    python scripts/prepare_neu_det.py --src path/to/NEU-DET --out data/neu_yolo

Works with both the original release (ANNOTATIONS/ + IMAGES/) and the Kaggle
layout (train/annotations, train/images, ...): it just searches --src
recursively for .xml files and the matching image. The split is stratified by
class so each defect type has the same val ratio; with only 300 images per
class a random split can easily leave one class under-represented.
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def voc_to_yolo(xml_path: Path) -> tuple[list[str], int, int]:
    root = ET.parse(xml_path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    lines = []
    for obj in root.iter("object"):
        name = obj.find("name").text.strip()
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue  # a handful of NEU boxes are degenerate
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines, w, h


def class_of(stem: str) -> str:
    # file names look like "rolled-in_scale_17" or "pitted_surface_119"
    return stem.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/neu_yolo"))
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = {p.stem: p for p in args.src.rglob("*") if p.suffix.lower() in IMG_EXT}
    by_class = defaultdict(list)
    for xml in sorted(args.src.rglob("*.xml")):
        if xml.stem in images:
            by_class[class_of(xml.stem)].append(xml)
        else:
            print(f"skip {xml.name}: no image found")

    rng = random.Random(args.seed)
    counts = defaultdict(lambda: [0, 0])
    for cls, xmls in sorted(by_class.items()):
        rng.shuffle(xmls)
        n_val = round(len(xmls) * args.val)
        for i, xml in enumerate(xmls):
            split = "val" if i < n_val else "train"
            lines, _, _ = voc_to_yolo(xml)
            img = images[xml.stem]
            (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, args.out / "images" / split / img.name)
            (args.out / "labels" / split / f"{xml.stem}.txt").write_text("\n".join(lines) + "\n")
            counts[cls][split == "val"] += 1

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    for cls, (tr, va) in sorted(counts.items()):
        print(f"{cls:16s} train {tr:4d}  val {va:4d}")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
