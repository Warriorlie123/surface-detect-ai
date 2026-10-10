from inspector import Detection
from inspector.inference import merge, tile_origins


def test_small_image_single_tile():
    assert tile_origins(150, 200, 0.2) == [0]
    assert tile_origins(200, 200, 0.2) == [0]


def test_tiles_cover_whole_axis():
    for length in (321, 640, 1000, 4096):
        starts = tile_origins(length, 320, 0.2)
        assert starts[0] == 0
        assert starts[-1] + 320 == length
        # no gaps between consecutive tiles
        assert all(b - a <= 320 for a, b in zip(starts, starts[1:]))


def test_merge_removes_duplicate_from_overlap():
    a = Detection(100, 100, 150, 150, "scratches", 0.9)
    b = Detection(102, 101, 151, 149, "scratches", 0.7)  # same defect seen by neighbouring tile
    c = Detection(102, 101, 151, 149, "patches", 0.6)  # different class, keep it
    kept = merge([a, b, c])
    assert a in kept and c in kept and b not in kept


def test_shift():
    d = Detection(0, 0, 10, 10, "x", 1.0).shifted(5, 7)
    assert (d.x1, d.y1, d.x2, d.y2) == (5, 7, 15, 17)
