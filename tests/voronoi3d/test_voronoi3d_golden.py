import numpy as np

from src.voronoi3d.voronoi import compute_voronoi_3d
from tests.voronoi3d.helpers_golden3d import (
    GOLDEN_ROOT,
    compute_metrics_3d,
    render_slice_png,
    save_metrics,
    save_png,
    load_metrics,
    assert_png_matches,
    update_mode,
)


def test_golden_single_seed():
    case_dir = GOLDEN_ROOT / "VG3D01_single_seed"
    png_path = case_dir / "slice_xy.png"
    json_path = case_dir / "metrics.json"

    size = (100.0, 60.0, 30.0)
    seeds = np.array([[50.0, 30.0, 15.0]], dtype=np.float64)

    d = compute_voronoi_3d(size_xyz=size, seeds_xyz=seeds, bounded=True, weld_decimals=6)

    metrics = compute_metrics_3d(d)
    png_bytes = render_slice_png(d, size, axis="z", value=15.0, draw_exposed_only=False)

    if update_mode():
        save_metrics(json_path, metrics)
        save_png(png_path, png_bytes)
        return

    ref = load_metrics(json_path)
    assert metrics["cell_count"] == ref["cell_count"]
    assert metrics["face_count"] == ref["face_count"]
    assert metrics["exposed_face_count"] == ref["exposed_face_count"]
    assert metrics["internal_face_count"] == ref["internal_face_count"]
    assert abs(metrics["avg_neighbors"] - ref["avg_neighbors"]) < 1e-6
    assert metrics["hash_vertices"] == ref["hash_vertices"]
    assert metrics["hash_faces"] == ref["hash_faces"]
    assert metrics["hash_adjacency_edges"] == ref["hash_adjacency_edges"]

    assert_png_matches(png_path, png_bytes, mean_abs_tol=1.0)


def test_golden_two_seeds_symmetric():
    case_dir = GOLDEN_ROOT / "VG3D02_two_seeds_symmetric"
    png_path = case_dir / "slice_xy.png"
    json_path = case_dir / "metrics.json"

    size = (100.0, 60.0, 30.0)
    seeds = np.array([[35.0, 30.0, 15.0], [65.0, 30.0, 15.0]], dtype=np.float64)

    d = compute_voronoi_3d(size_xyz=size, seeds_xyz=seeds, bounded=True, weld_decimals=6)

    metrics = compute_metrics_3d(d)
    png_bytes = render_slice_png(d, size, axis="z", value=15.0, draw_exposed_only=False)

    if update_mode():
        save_metrics(json_path, metrics)
        save_png(png_path, png_bytes)
        return

    ref = load_metrics(json_path)
    assert metrics["cell_count"] == ref["cell_count"]
    assert metrics["face_count"] == ref["face_count"]
    assert metrics["exposed_face_count"] == ref["exposed_face_count"]
    assert metrics["internal_face_count"] == ref["internal_face_count"]
    assert abs(metrics["avg_neighbors"] - ref["avg_neighbors"]) < 0.05
    assert metrics["hash_vertices"] == ref["hash_vertices"]
    assert metrics["hash_faces"] == ref["hash_faces"]
    assert metrics["hash_adjacency_edges"] == ref["hash_adjacency_edges"]

    assert_png_matches(png_path, png_bytes, mean_abs_tol=1.5)


def test_golden_random_200():
    case_dir = GOLDEN_ROOT / "VG3D03_random_200"
    png_path = case_dir / "slice_xy.png"
    json_path = case_dir / "metrics.json"

    size = (120.0, 80.0, 40.0)

    rng = np.random.default_rng(123)
    seeds = np.empty((200, 3), dtype=np.float64)
    seeds[:, 0] = rng.random(200) * size[0]
    seeds[:, 1] = rng.random(200) * size[1]
    seeds[:, 2] = rng.random(200) * size[2]

    d = compute_voronoi_3d(size_xyz=size, seeds_xyz=seeds, bounded=True, weld_decimals=6)

    metrics = compute_metrics_3d(d)
    png_bytes = render_slice_png(d, size, axis="z", value=size[2] * 0.5, draw_exposed_only=False)

    if update_mode():
        save_metrics(json_path, metrics)
        save_png(png_path, png_bytes)
        return

    ref = load_metrics(json_path)
    assert metrics["cell_count"] == ref["cell_count"]
    assert metrics["face_count"] == ref["face_count"]
    assert metrics["exposed_face_count"] == ref["exposed_face_count"]
    assert metrics["internal_face_count"] == ref["internal_face_count"]
    assert abs(metrics["avg_neighbors"] - ref["avg_neighbors"]) < 0.15
    assert metrics["hash_vertices"] == ref["hash_vertices"]
    assert metrics["hash_faces"] == ref["hash_faces"]
    assert metrics["hash_adjacency_edges"] == ref["hash_adjacency_edges"]

    # slightly looser tol because many segments
    assert_png_matches(png_path, png_bytes, mean_abs_tol=2.0)
