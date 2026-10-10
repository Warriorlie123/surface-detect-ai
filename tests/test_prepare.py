import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from prepare_neu_det import class_of, voc_to_yolo  # noqa: E402

XML = """<annotation><size><width>200</width><height>200</height><depth>1</depth></size>
<object><name>scratches</name><bndbox><xmin>50</xmin><ymin>0</ymin><xmax>150</xmax><ymax>200</ymax></bndbox></object>
<object><name>scratches</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>10</xmax><ymax>30</ymax></bndbox></object>
</annotation>"""


def test_voc_to_yolo(tmp_path):
    p = tmp_path / "scratches_1.xml"
    p.write_text(XML)
    lines, w, h = voc_to_yolo(p)
    assert (w, h) == (200, 200)
    assert lines == ["5 0.500000 0.500000 0.500000 1.000000"]  # degenerate box dropped


def test_class_of():
    assert class_of("rolled-in_scale_17") == "rolled-in_scale"
    assert class_of("pitted_surface_119") == "pitted_surface"
    assert class_of("crazing_1") == "crazing"
