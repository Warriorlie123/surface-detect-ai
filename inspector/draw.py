from PIL import Image, ImageDraw, ImageFont

from .detection import Detection

# one fixed colour per class so the same defect type looks the same everywhere
COLORS = {
    "crazing": (230, 159, 0),
    "inclusion": (213, 94, 0),
    "patches": (0, 114, 178),
    "pitted_surface": (204, 121, 167),
    "rolled-in_scale": (0, 158, 115),
    "scratches": (240, 228, 66),
}


def draw_detections(
    image: Image.Image, dets: list[Detection], dashed: set[int] | None = None, show_conf: bool = True
) -> Image.Image:
    """Draw boxes on a copy of the image. Indices in `dashed` are drawn thin
    (used for low-confidence detections that only trigger a review).
    show_conf=False labels boxes with the class only (for annotator boxes)."""
    out = image.convert("RGB").copy()
    # NEU crops are only 200 px wide; upscale so labels stay readable
    scale = max(1, 600 // max(out.size))
    if scale > 1:
        out = out.resize((out.width * scale, out.height * scale), Image.NEAREST)

    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=max(12, min(out.size) // 40))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    dashed = dashed or set()
    lw = max(2, min(out.size) // 250)
    for i, det in enumerate(dets):
        color = COLORS.get(det.label, (255, 0, 0))
        box = [det.x1 * scale, det.y1 * scale, det.x2 * scale, det.y2 * scale]
        d.rectangle(box, outline=color, width=1 if i in dashed else lw)
        text = f"{det.label} {det.conf:.2f}" if show_conf else det.label
        tb = d.textbbox((box[0], box[1]), text, font=font)
        ty = box[1] - (tb[3] - tb[1]) - 4
        if ty < 0:
            ty = box[1]
        d.rectangle([box[0], ty, box[0] + (tb[2] - tb[0]) + 4, ty + (tb[3] - tb[1]) + 4], fill=color)
        d.text((box[0] + 2, ty), text, fill=(0, 0, 0), font=font)
    return out
