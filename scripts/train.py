"""Train a detector from one of the configs in configs/.

    python scripts/train.py --config configs/yolov8s_aug.yaml --data data/neu_yolo/data.yaml

Both models were trained on Google Colab. The baseline took ~10 min, the yolov8s
run ~36 min.
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    model = YOLO(cfg.pop("model"))
    model.train(data=str(args.data), device=args.device, project="runs", name=args.config.stem, **cfg)


if __name__ == "__main__":
    main()
