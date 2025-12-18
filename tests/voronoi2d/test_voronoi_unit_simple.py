import numpy as np
from src.voronoi2d.geometry import plane_from_polygon, project_xyz_to_uv
from src.voronoi2d.voronoi import compute_voronoi_2d


def _square_polygon_xyz(size=10.0):
    return np.array([
        [0.0, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, size, 0.0],
        [0.0, size, 0.0],
    ], dtype=np.float64)


def test_voronoi_two_points_split():
    poly_xyz = _square_polygon_xyz(10.0)
    origin, u, v, n = plane_from_polygon(poly_xyz)
    poly_uv = project_xyz_to_uv(poly_xyz, origin, u, v)

    seeds_uv = np.array([
        [3.0, 5.0],
        [7.0, 5.0],
    ], dtype=np.float64)

    d = compute_voronoi_2d(poly_uv, poly_xyz, seeds_uv, origin, u, v, bounded=True)

    # Should have 2 cells for the 2 seeds
    ids = sorted([c.index for c in d.cells])
    assert ids == [0, 1]

    # There should be at least one internal edge with 2 cells
    internal = [e for e in d.edges if len(e.cells) == 2]
    assert len(internal) >= 1

    # Each cell should have at least one neighbor (the other)
    c0 = next(c for c in d.cells if c.index == 0)
    c1 = next(c for c in d.cells if c.index == 1)
    assert 1 in c0.neighbors
    assert 0 in c1.neighbors


def test_voronoi_center_has_four_neighbors_in_symmetric_setup():
    poly_xyz = _square_polygon_xyz(10.0)
    origin, u, v, n = plane_from_polygon(poly_xyz)
    poly_uv = project_xyz_to_uv(poly_xyz, origin, u, v)

    # 5 points: center + 4 around it
    seeds_uv = np.array([
        [5.0, 5.0],   # center (id 0)
        [2.5, 5.0],   # left
        [7.5, 5.0],   # right
        [5.0, 2.5],   # bottom
        [5.0, 7.5],   # top
    ], dtype=np.float64)

    d = compute_voronoi_2d(poly_uv, poly_xyz, seeds_uv, origin, u, v, bounded=True)

    center = next(c for c in d.cells if c.index == 0)

    # With reflections bounding, all 5 should exist
    assert set([c.index for c in d.cells]) == {0, 1, 2, 3, 4}

    # center should be adjacent to the 4 surrounding points
    assert set(center.neighbors) == {1, 2, 3, 4}
