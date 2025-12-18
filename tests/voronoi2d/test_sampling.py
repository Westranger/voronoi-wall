import numpy as np
from shapely.geometry import Polygon, Point
from src.voronoi2d.sampling import sample_points_in_polygon


def test_sampling_points_inside_polygon():
    rng = np.random.default_rng(123)
    poly_uv = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    pts = sample_points_in_polygon(poly_uv, n_points=100, rng=rng)

    poly = Polygon(poly_uv)
    assert len(pts) == 100
    for p in pts:
        assert poly.contains(Point(float(p[0]), float(p[1])))


def test_sampling_count_from_target_area():
    rng = np.random.default_rng(0)
    poly_uv = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)  # area=100
    pts = sample_points_in_polygon(poly_uv, target_area_mm2=25.0, rng=rng)
    # int(100/25)=4
    assert len(pts) == 4


def test_sampling_is_deterministic_with_seed():
    poly_uv = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)

    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    a = sample_points_in_polygon(poly_uv, n_points=20, rng=rng1)
    b = sample_points_in_polygon(poly_uv, n_points=20, rng=rng2)

    assert np.allclose(a, b)
