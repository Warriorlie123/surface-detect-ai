"""Run the full inspection (detection + decision rules) on a folder of images
without the UI and write one CSV row per image.

    python scripts/inspect_folder.py sample_images --field-mm 50 --out report.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspector import Calibration, InspectionPolicy, decide  # noqa: E402
from inspector.inference import WEIGHTS, load_model, predict  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--model", default="yolov8s_aug", choices=list(WEIGHTS))
    ap.add_argument("--field-mm", type=float, default=50.0, help="visible surface width in mm")
    ap.add_argument("--review-conf", type=float, default=0.25)
    ap.add_argument("--reject-conf", type=float, default=0.50)
    ap.add_argument("--max-length-mm", type=float, default=10.0)
    ap.add_argument("--max-coverage", type=float, default=5.0)
    ap.add_argument("--zero-tolerance", nargs="*", default=["inclusion"])
    ap.add_argument("--tile", type=int, default=None, help="tile size in px for large images")
    ap.add_argument("--out", type=Path, default=Path("inspection_report.csv"))
    args = ap.parse_args()

    policy = InspectionPolicy(
        reject_conf=args.reject_conf,
        review_conf=args.review_conf,
        max_length_mm=args.max_length_mm,
        max_coverage_pct=args.max_coverage,
        zero_tolerance=set(args.zero_tolerance),
    )
    model = load_model(args.model)
    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in EXT)
    if not files:
        sys.exit(f"no images in {args.folder}")

    tally = {"PASS": 0, "REVIEW": 0, "REJECT": 0}
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "verdict", "reasons", "confirmed", "uncertain", "classes", "latency_ms"])
        for p in files:
            img = Image.open(p)
            t = time.perf_counter()
            dets = predict(model, img, conf=policy.review_conf, tile=args.tile)
            ms = (time.perf_counter() - t) * 1000
            dec = decide(dets, Calibration(img.width, img.height, args.field_mm), policy)
            tally[dec.verdict.value] += 1
            classes = ",".join(sorted({d.label for d in dets}))
            w.writerow(
                [
                    p.name,
                    dec.verdict.value,
                    "; ".join(dec.reasons),
                    len(dec.confirmed),
                    len(dec.uncertain),
                    classes,
                    f"{ms:.1f}",
                ]
            )
            print(f"{p.name:28s} {dec.verdict.value:7s} {'; '.join(dec.reasons)}")

    print(f"\n{len(files)} images: " + ", ".join(f"{k} {v}" for k, v in tally.items()) + f"  ->  {args.out}")


if __name__ == "__main__":
    main()
