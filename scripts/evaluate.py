"""Evaluate one or more weight files on the same validation split.

    python scripts/evaluate.py --data data/neu_yolo/data.yaml
    python scripts/evaluate.py --data ... --weights weights/yolov8s_aug.pt

Prints overall and per-class AP and writes results to eval/<weights>.json.
Important: compare models only on a split neither of them was trained on.
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

DEFAULT = ["weights/yolov8n_baseline.pt", "weights/yolov8s_aug.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", nargs="+", default=DEFAULT)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=Path("eval"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    for w in args.weights:
        model = YOLO(w)
        m = model.val(data=args.data, split=args.split, plots=False, verbose=False)
        per_class = {
            model.names[c]: {"AP50": float(m.box.ap50[i]), "AP50-95": float(m.box.ap[i])}
            for i, c in enumerate(m.box.ap_class_index)
        }
        result = {
            "weights": w,
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
            "mAP50": float(m.box.map50),
            "mAP50-95": float(m.box.map),
            "per_class": per_class,
        }
        (args.out / f"{Path(w).stem}.json").write_text(json.dumps(result, indent=2))

        print(
            f"\n{w}: P {result['precision']:.3f}  R {result['recall']:.3f}  "
            f"mAP50 {result['mAP50']:.3f}  mAP50-95 {result['mAP50-95']:.3f}"
        )
        for name, v in per_class.items():
            print(f"  {name:16s} AP50 {v['AP50']:.3f}  AP50-95 {v['AP50-95']:.3f}")


if __name__ == "__main__":
    main()
