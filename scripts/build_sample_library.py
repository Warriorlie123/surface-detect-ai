"""Copy a NEU-DET split into sample_images/neu_det/ so the app can browse it.

    python scripts/build_sample_library.py --src path/to/NEU-DET/validation

Expects the Kaggle layout (images/<class>/*.jpg and annotations/*.xml). Images
are copied into one folder per class, and all bounding boxes are written to a
single labels.json so the app can draw the annotator's boxes next to the
model's predictions without shipping hundreds of XML files.
"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def read_boxes(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    boxes = []
    for obj in root.iter("object"):
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue  # a handful of NEU boxes are degenerate
        boxes.append({"label": obj.find("name").text.strip(), "box": [x1, y1, x2, y2]})
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="a NEU-DET split, e.g. NEU-DET/validation")
    ap.add_argument("--out", type=Path, default=Path("sample_images/neu_det"))
    args = ap.parse_args()

    labels = {}
    for cls in CLASSES:
        (args.out / cls).mkdir(parents=True, exist_ok=True)
        for img in sorted((args.src / "images" / cls).glob("*.jpg")):
            xml = args.src / "annotations" / f"{img.stem}.xml"
            if not xml.exists():
                print(f"skip {img.name}: no annotation")
                continue
            shutil.copy2(img, args.out / cls / img.name)
            labels[img.name] = read_boxes(xml)

    (args.out / "labels.json").write_text(json.dumps(labels, indent=0, sort_keys=True) + "\n")
    per_class = {c: sum(1 for n in labels if n.rsplit("_", 1)[0] == c) for c in CLASSES}
    print(", ".join(f"{c} {n}" for c, n in per_class.items()), f"-> {args.out}")


if __name__ == "__main__":
    main()
