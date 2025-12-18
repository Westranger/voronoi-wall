import numpy as np

from src.voronoi3d.sampling import sample_points_in_box, sample_points_in_box_by_target_volume


def test_sample_points_in_box_bounds():
    rng = np.random.default_rng(0)
    pts = sample_points_in_box((10.0, 20.0, 30.0), 500, rng)
    assert pts.shape == (500, 3)
    assert np.all(pts[:, 0] >= 0.0) and np.all(pts[:, 0] <= 10.0)
    assert np.all(pts[:, 1] >= 0.0) and np.all(pts[:, 1] <= 20.0)
    assert np.all(pts[:, 2] >= 0.0) and np.all(pts[:, 2] <= 30.0)


def test_sample_points_by_target_volume():
    rng = np.random.default_rng(0)
    size = (10.0, 10.0, 10.0)  # volume 1000
    pts = sample_points_in_box_by_target_volume(size, target_cell_volume_mm3=10.0, rng=rng)
    # expect around 100 points
    assert 50 <= len(pts) <= 200
    assert pts.shape[1] == 3
