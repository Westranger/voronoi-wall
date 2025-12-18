import numpy as np
from src.voronoi2d.geometry import plane_from_polygon, project_xyz_to_uv, lift_uv_to_xyz


def test_plane_basis_is_orthonormal():
    poly = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
    ], dtype=np.float64)

    origin, u, v, n = plane_from_polygon(poly)

    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.isclose(np.linalg.norm(n), 1.0)

    assert abs(u @ v) < 1e-9
    assert abs(u @ n) < 1e-9
    assert abs(v @ n) < 1e-9


def test_project_lift_roundtrip():
    poly = np.array([
        [3.0, 2.0, 1.0],
        [13.0, 2.0, 1.0],
        [13.0, 7.0, 1.0],
        [3.0, 7.0, 1.0],
    ], dtype=np.float64)

    origin, u, v, n = plane_from_polygon(poly)

    pts = np.array([
        [5.0, 3.0, 1.0],
        [12.0, 6.0, 1.0],
        [9.0, 4.0, 1.0],
    ], dtype=np.float64)

    uv = project_xyz_to_uv(pts, origin, u, v)
    back = lift_uv_to_xyz(uv, origin, u, v)

    assert np.allclose(back, pts, atol=1e-9)


def test_lifted_points_are_planar():
    poly = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 1.0],
        [10.0, 5.0, 2.0],
        [0.0, 5.0, 1.0],
    ], dtype=np.float64)

    origin, u, v, n = plane_from_polygon(poly)

    uv = project_xyz_to_uv(poly, origin, u, v)
    back = lift_uv_to_xyz(uv, origin, u, v)

    # Check plane equation
    d = (back - origin) @ n
    assert np.allclose(d, 0.0, atol=1e-6)
