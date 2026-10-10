"""CPU latency: PyTorch vs ONNX Runtime for both models.

    python scripts/benchmark.py            # 640 px input, 50 runs each
    python scripts/benchmark.py --imgsz 320

Exports each model to ONNX next to the .pt file (skipped if it already
exists). Timings are for model.predict() end to end, so they include
pre/post-processing and NMS, which is what the app actually pays.
"""

import argparse
import platform
import statistics
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

WEIGHTS = ["weights/yolov8n_baseline.pt", "weights/yolov8s_aug.pt"]


def time_model(model, img, imgsz, runs, warmup=5):
    for _ in range(warmup):
        model.predict(img, imgsz=imgsz, verbose=False)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        model.predict(img, imgsz=imgsz, verbose=False)
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), sorted(ts)[int(0.9 * len(ts)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--image", default="sample_images/scratches_25.jpg")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    print(f"{platform.processor() or platform.machine()}, imgsz={args.imgsz}, {args.runs} runs\n")
    print(f"{'model':22s} {'backend':8s} {'median ms':>10s} {'p90 ms':>8s}")

    for w in WEIGHTS:
        onnx = Path(w).with_name(f"{Path(w).stem}_{args.imgsz}.onnx")
        if not onnx.exists():
            out = YOLO(w).export(format="onnx", imgsz=args.imgsz, simplify=True, verbose=False)
            Path(out).rename(onnx)
        for backend, path in [("torch", w), ("onnx", str(onnx))]:
            med, p90 = time_model(YOLO(path, task="detect"), img, args.imgsz, args.runs)
            print(f"{Path(w).stem:22s} {backend:8s} {med:10.1f} {p90:8.1f}")


if __name__ == "__main__":
    main()
