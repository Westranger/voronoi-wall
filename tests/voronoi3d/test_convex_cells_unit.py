import numpy as np

from src.voronoi3d.convex_cells import voronoi_cell_halfspaces_in_box


def test_two_seeds_split_volume_symmetric():
    size = (10.0, 10.0, 10.0)  # volume 1000
    seeds = np.array([
        [2.5, 5.0, 5.0],
        [7.5, 5.0, 5.0],
    ], dtype=np.float64)

    c0 = voronoi_cell_halfspaces_in_box(seeds, 0, size)
    c1 = voronoi_cell_halfspaces_in_box(seeds, 1, size)

    v0 = c0.volume()
    v1 = c1.volume()

    assert v0 > 0
    assert v1 > 0

    # symmetric => nearly equal
    assert abs(v0 - v1) / max(v0, v1) < 0.02

    # sum approx box volume
    assert abs((v0 + v1) - 1000.0) / 1000.0 < 0.02


def test_single_seed_is_full_box_volume():
    size = (10.0, 10.0, 10.0)
    seeds = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
    c0 = voronoi_cell_halfspaces_in_box(seeds, 0, size)
    v0 = c0.volume()
    assert abs(v0 - 1000.0) / 1000.0 < 0.02
