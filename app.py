import hashlib
import math
import random
import time
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from inspector import Calibration, InspectionPolicy, Verdict, decide, defect_coverage, measure
from inspector.draw import draw_detections
from inspector.groundtruth import compare, load_labels
from inspector.inference import ROOT, WEIGHTS, load_model, predict

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SAMPLES = sorted((ROOT / "sample_images").glob("*.jpg"))

# The 360 NEU-DET validation images (60 per class), browsable in the app.
# Built with scripts/build_sample_library.py.
LIBRARY_DIR = ROOT / "sample_images" / "neu_det"
LIBRARY = {c: sorted((LIBRARY_DIR / c).glob("*.jpg"), key=lambda p: int(p.stem.rsplit("_", 1)[1])) for c in CLASSES}
LIBRARY_SIZE = sum(len(v) for v in LIBRARY.values())
GROUND_TRUTH = load_labels(LIBRARY_DIR / "labels.json")
PAGE_SIZE = 12
VERDICT_STYLE = {Verdict.PASS: st.success, Verdict.REVIEW: st.warning, Verdict.REJECT: st.error}

st.set_page_config(page_title="Steel Surface Inspection", layout="wide")
st.title("Steel Surface Defect Inspection")
st.caption(
    "YOLOv8 trained on NEU-DET (hot-rolled steel strip, 6 defect classes). "
    "Upload one image for a detailed view or several for a batch run."
)

if "log" not in st.session_state:
    st.session_state.log = {}  # keyed by image hash so re-running doesn't duplicate rows
if "sample" not in st.session_state:
    st.session_state.sample = []  # list of Paths picked from the bundled samples


def pretty(name: str) -> str:
    return name.replace("_", " ").replace("-", " ")


# ---------------- sidebar ----------------
with st.sidebar:
    model_name = st.selectbox(
        "Model", list(WEIGHTS), help="yolov8s_aug is the main model, yolov8n_baseline the first run."
    )

    st.subheader("Decision rules")
    review_conf, reject_conf = st.slider(
        "Review / reject confidence",
        0.05,
        0.95,
        (0.25, 0.50),
        0.05,
        help="Below the left handle: ignored. Between handles: sent to manual review. Above the right handle: confirmed defect.",
    )
    max_len = st.number_input("Max allowed defect length (mm)", 0.0, 500.0, 10.0, 0.5)
    max_cov = st.number_input("Max allowed defect coverage (%)", 0.0, 100.0, 5.0, 0.5)
    zero_tol = st.multiselect("Zero-tolerance classes", CLASSES, default=["inclusion"])

    st.subheader("Calibration")
    field_mm = st.number_input(
        "Visible surface width (mm)",
        1.0,
        5000.0,
        50.0,
        1.0,
        help="Real width of the strip area the camera sees. Used to convert pixels to mm.",
    )

    st.subheader("Large images")
    tiling = st.toggle(
        "Tiled inference", value=True, help="Split images larger than the tile size into overlapping crops."
    )
    tile_px = st.select_slider("Tile size (px)", [200, 320, 480, 640], value=320, disabled=not tiling)

policy = InspectionPolicy(
    reject_conf=reject_conf,
    review_conf=review_conf,
    max_length_mm=max_len,
    max_coverage_pct=max_cov,
    zero_tolerance=set(zero_tol),
)


@st.cache_data(max_entries=500, show_spinner=False)
def run_model(name: str, data: bytes, conf: float, tile: int | None):
    # Cached so that moving a rule slider doesn't re-run the network on every
    # image in a batch. Only the decision depends on the rules, and that's cheap.
    t0 = time.perf_counter()
    dets = predict(load_model(name), open_image(data), conf=conf, tile=tile)
    return dets, (time.perf_counter() - t0) * 1000


def inspect(name: str, data: bytes):
    image = open_image(data)
    cal = Calibration(image.width, image.height, field_mm)
    dets, ms = run_model(name, data, policy.review_conf, tile_px if tiling else None)
    return image, dets, cal, decide(dets, cal, policy), ms


def open_image(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data))


def log_result(name, image_bytes, dets, decision, ms):
    key = hashlib.md5(image_bytes).hexdigest()
    st.session_state.log[key] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": name,
        "model": model_name,
        "verdict": decision.verdict.value,
        "reasons": "; ".join(decision.reasons),
        "defects": len(decision.confirmed),
        "uncertain": len(decision.uncertain),
        "classes": ", ".join(sorted({d.label for d in dets})) or "-",
        "max_conf": round(max((d.conf for d in dets), default=0.0), 3),
        "latency_ms": round(ms, 1),
    }


def defect_table(dets, cal):
    rows = []
    for d in sorted(dets, key=lambda d: -d.conf):
        m = measure(d, cal)
        rows.append(
            {
                "class": m.label,
                "confidence": round(m.conf, 3),
                "status": "confirmed" if d.conf >= policy.reject_conf else "review",
                "length (mm)": round(m.length_mm, 2),
                "width (mm)": round(m.width_mm, 2),
                "area (mm²)": round(m.area_mm2, 2),
            }
        )
    return pd.DataFrame(rows)


# ---------------- input ----------------
uploads = st.file_uploader(
    "Surface images", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True
)

with st.expander("No images? Try samples from the NEU-DET dataset", expanded=not uploads):
    quick, browse = st.tabs(["Quick picks", f"Browse all {LIBRARY_SIZE} validation images"])

    with quick:
        cols = st.columns(len(SAMPLES))
        for col, path in zip(cols, SAMPLES):
            col.image(str(path), width="stretch")
            if col.button(pretty(path.stem.rsplit("_", 1)[0]), key=path.name, width="stretch"):
                st.session_state.sample = [path]

    with browse:
        if not LIBRARY_SIZE:
            st.info(
                "Sample library not found. Build it with `python scripts/build_sample_library.py --src NEU-DET/validation`."
            )
        else:
            st.caption(
                "Pick any image to inspect it. The annotator's boxes from NEU-DET are shown next to the "
                "model's prediction so you can check it yourself."
            )
            a, b = st.columns([2, 3])
            cls = a.selectbox("Defect class", CLASSES, format_func=pretty, key="lib_class")
            paths = LIBRARY[cls]
            pages = max(1, math.ceil(len(paths) / PAGE_SIZE))
            page = b.pills("Page", list(range(1, pages + 1)), default=1, key=f"lib_page_{cls}") or 1

            grid = st.columns(6)
            for i, path in enumerate(paths[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]):
                col = grid[i % 6]
                col.image(str(path), width="stretch")
                n_boxes = len(GROUND_TRUTH.get(path.name, []))
                label = f"#{path.stem.rsplit('_', 1)[1]} · {n_boxes} box{'es' if n_boxes != 1 else ''}"
                if col.button(label, key=f"lib_{path.name}", width="stretch"):
                    st.session_state.sample = [path]

            r1, r2, r3 = st.columns(3)
            if r1.button(f"Random {pretty(cls)}", width="stretch"):
                st.session_state.sample = [random.choice(paths)]
            if r2.button("Random from any class", width="stretch"):
                st.session_state.sample = [random.choice([p for ps in LIBRARY.values() for p in ps])]
            if r3.button(f"Run all {len(paths)} {pretty(cls)} images", width="stretch"):
                st.session_state.sample = list(paths)

if uploads:
    st.session_state.sample = []
    inputs = [(f.name, f.getvalue()) for f in uploads]
else:
    inputs = [(p.name, p.read_bytes()) for p in st.session_state.sample]


# ---------------- single image ----------------
if len(inputs) == 1:
    name, data = inputs[0]
    image, dets, cal, decision, ms = inspect(model_name, data)
    log_result(name, data, dets, decision, ms)

    reasons = ": " + "; ".join(decision.reasons) if decision.reasons else ""
    VERDICT_STYLE[decision.verdict](f"**{decision.verdict.value}**{reasons}")

    truth = GROUND_TRUTH.get(name)
    left, right = st.columns(2)
    if truth is not None:
        left.image(
            draw_detections(image, truth, show_conf=False),
            caption=f"{name}: annotator boxes from NEU-DET ({len(truth)})",
            width="stretch",
        )
    else:
        left.image(image, caption=f"{name} ({image.width}×{image.height} px)", width="stretch")
    uncertain_idx = {i for i, d in enumerate(dets) if d.conf < policy.reject_conf}
    right.image(
        draw_detections(image, dets, uncertain_idx),
        caption="Model prediction. Thin boxes = below reject threshold",
        width="stretch",
    )

    if truth is not None:
        cmp = compare(dets, truth)
        missed = ", ".join(sorted({d.label for d in cmp.missed})) or "none"
        st.info(
            f"**Versus the annotator:** found {cmp.found} of {cmp.total} annotated box(es) "
            f"(same class, IoU ≥ 0.5). Missed: {missed}. Extra predictions: {len(cmp.extra)}. "
            "NEU boxes are not always consistent (see README), so an extra box can still be a real defect."
        )

    area, cov = defect_coverage(decision.confirmed, cal)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inference", f"{ms:.0f} ms")
    c2.metric("Confirmed defects", len(decision.confirmed))
    c3.metric("Defect area", f"{area:.1f} mm²")
    c4.metric("Coverage", f"{cov:.2f} %")

    if dets:
        st.dataframe(defect_table(dets, cal), width="stretch", hide_index=True)

    with st.expander("Compare with the other model"):
        other_name = next(n for n in WEIGHTS if n != model_name)
        _, other_dets, _, other_decision, other_ms = inspect(other_name, data)
        a, b = st.columns(2)
        for col, n, ds, dec, t in [
            (a, model_name, dets, decision, ms),
            (b, other_name, other_dets, other_decision, other_ms),
        ]:
            col.markdown(f"**{n}**: {dec.verdict.value}, {len(ds)} box(es), {t:.0f} ms")
            col.image(draw_detections(image, ds), width="stretch")
        st.caption(
            "On the validation set the two models are within 0.01 mAP50 of each other (see README). "
            "Differences on a single image are mostly noise around the confidence threshold."
        )

# ---------------- batch ----------------
elif len(inputs) > 1:
    progress = st.progress(0.0, text="Inspecting...")
    rows = []
    for i, (name, data) in enumerate(inputs):
        image, dets, cal, decision, ms = inspect(model_name, data)
        log_result(name, data, dets, decision, ms)
        rows.append((name, image, dets, decision))
        progress.progress((i + 1) / len(inputs), text=f"{i + 1}/{len(inputs)}")
    progress.empty()

    gt_rows = [(compare(r[2], GROUND_TRUTH[r[0]])) for r in rows if r[0] in GROUND_TRUTH]
    if gt_rows:
        found, total = sum(c.found for c in gt_rows), sum(c.total for c in gt_rows)
        st.info(
            f"**Versus the annotator** on {len(gt_rows)} NEU-DET image(s): found {found} of {total} annotated boxes "
            f"({found / max(total, 1):.0%}, same class, IoU ≥ 0.5), {sum(len(c.extra) for c in gt_rows)} extra predictions."
        )

    counts = pd.Series([r[3].verdict.value for r in rows]).value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Pass", int(counts.get("PASS", 0)))
    c2.metric("Review", int(counts.get("REVIEW", 0)))
    c3.metric("Reject", int(counts.get("REJECT", 0)))

    show = st.radio("Show", ["REJECT", "REVIEW", "PASS", "all"], horizontal=True)
    shown = [r for r in rows if show == "all" or r[3].verdict.value == show]
    grid = st.columns(4)
    for i, (name, image, dets, decision) in enumerate(shown):
        uncertain_idx = {j for j, d in enumerate(dets) if d.conf < policy.reject_conf}
        grid[i % 4].image(
            draw_detections(image, dets, uncertain_idx), caption=f"{name} · {decision.verdict.value}", width="stretch"
        )


# ---------------- log ----------------
st.divider()
st.subheader("Inspection log")
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log.values())
    st.dataframe(df, width="stretch", hide_index=True)
    a, b = st.columns([1, 5])
    a.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8-sig"),  # BOM so Excel opens it with the right encoding
        file_name=f"inspection_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )
    if b.button("Clear log"):
        st.session_state.log = {}
        st.rerun()
else:
    st.caption("Every inspected image is added here automatically.")
