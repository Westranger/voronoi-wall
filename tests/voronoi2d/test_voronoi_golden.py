import matplotlib

matplotlib.use("Agg")
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

from src.voronoi2d.geometry import plane_from_polygon, project_xyz_to_uv
from src.voronoi2d.sampling import sample_points_in_polygon
from src.voronoi2d.voronoi import compute_voronoi_2d

from tests.voronoi2d.helpers_golden import (
    GOLDEN_ROOT,
    compute_metrics,
    save_metrics,
    load_metrics,
    save_png,
    assert_png_matches,
    update_mode,
)


def _edge_endpoints(edge):
    # common attribute patterns
    for A, B in [("a", "b"), ("va", "vb"), ("v0", "v1"), ("i", "j"), ("u", "v")]:
        if hasattr(edge, A) and hasattr(edge, B):
            return int(getattr(edge, A)), int(getattr(edge, B))

    # sometimes stored as tuple/list
    if hasattr(edge, "vertices"):
        vv = getattr(edge, "vertices")
        if isinstance(vv, (tuple, list)) and len(vv) == 2:
            return int(vv[0]), int(vv[1])

    raise TypeError(f"Don't know how to read endpoints from edge type: {type(edge)}")


def _render_uv_png(d, poly_uv: np.ndarray, img_size=(900, 600), padding_px=10) -> bytes:
    W, H = img_size
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Stable framing: polygon bounds
    minx, miny = poly_uv.min(axis=0)
    maxx, maxy = poly_uv.max(axis=0)
    spanx = max(maxx - minx, 1e-9)
    spany = max(maxy - miny, 1e-9)

    sx = (W - 2 * padding_px) / spanx
    sy = (H - 2 * padding_px) / spany
    s = min(sx, sy)

    def map_pt(p):
        x = padding_px + (float(p[0]) - float(minx)) * s
        y = padding_px + (float(p[1]) - float(miny)) * s
        return (x, y)

    # draw polygon outline
    poly_pts = [map_pt(p) for p in poly_uv]
    draw.line(poly_pts + [poly_pts[0]], fill=(0, 0, 0), width=3)

    # draw all Voronoi edges from topology
    V = d.vertices_uv  # or d.vertices depending on your diagram
    for e in d.edges:
        a, b = _edge_endpoints(e)
        p = map_pt(V[a])
        q = map_pt(V[b])
        draw.line([p, q], fill=(0, 0, 0), width=2)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tilted_rect_xyz(w=100.0, h=60.0):
    # rectangle in 3D, tilted by embedding in a simple plane (z depends on x,y)
    return np.array([
        [0.0, 0.0, 0.0],
        [w, 0.0, 3.0],
        [w, h, 6.0],
        [0.0, h, 3.0],
    ], dtype=np.float64)


def test_golden_single_region():
    case_dir = GOLDEN_ROOT / "VG01_single_region"
    png_path = case_dir / "preview_uv.png"
    json_path = case_dir / "metrics.json"

    poly_xyz = _tilted_rect_xyz(100.0, 60.0)
    origin, u, v, n = plane_from_polygon(poly_xyz)
    poly_uv = project_xyz_to_uv(poly_xyz, origin, u, v)

    rng = np.random.default_rng(42)
    seeds_uv = sample_points_in_polygon(poly_uv, target_area_mm2=30.0, rng=rng)  # ~200 points
    d = compute_voronoi_2d(poly_uv, poly_xyz, seeds_uv, origin, u, v, bounded=True)

    metrics = compute_metrics(d)
    png_bytes = _render_uv_png(d, poly_uv)

    if update_mode():
        save_metrics(json_path, metrics)
        save_png(png_path, png_bytes)
        return

    # compare json
    ref = load_metrics(json_path)
    assert metrics["cell_count"] == ref["cell_count"]
    assert metrics["edge_count"] == ref["edge_count"]
    assert abs(metrics["avg_neighbors"] - ref["avg_neighbors"]) < 0.05
    assert metrics["hash_vertices_uv"] == ref["hash_vertices_uv"]
    assert metrics["hash_edges"] == ref["hash_edges"]

    # compare png with tolerance
    assert_png_matches(png_path, png_bytes, mean_abs_tol=1.5)


def test_golden_two_regions_density_change():
    case_dir = GOLDEN_ROOT / "VG02_two_regions"
    png_path = case_dir / "preview_uv.png"
    json_path = case_dir / "metrics.json"

    poly_xyz = _tilted_rect_xyz(100.0, 60.0)
    origin, u, v, n = plane_from_polygon(poly_xyz)
    poly_uv = project_xyz_to_uv(poly_xyz, origin, u, v)

    # define two half-rectangles in UV
    minx, miny = poly_uv.min(axis=0)
    maxx, maxy = poly_uv.max(axis=0)
    midx = 0.5 * (minx + maxx)

    left = np.array([[minx, miny], [midx, miny], [midx, maxy], [minx, maxy]], dtype=np.float64)
    right = np.array([[midx, miny], [maxx, miny], [maxx, maxy], [midx, maxy]], dtype=np.float64)

    rngL = np.random.default_rng(10)
    rngR = np.random.default_rng(11)

    seeds_left = sample_points_in_polygon(left, target_area_mm2=20.0, rng=rngL)  # denser
    seeds_right = sample_points_in_polygon(right, target_area_mm2=80.0, rng=rngR)  # coarser
    seeds_uv = np.vstack([seeds_left, seeds_right])

    d = compute_voronoi_2d(poly_uv, poly_xyz, seeds_uv, origin, u, v, bounded=True)

    metrics = compute_metrics(d)
    png_bytes = _render_uv_png(d, poly_uv)

    if update_mode():
        save_metrics(json_path, metrics)
        save_png(png_path, png_bytes)
        return

    ref = load_metrics(json_path)
    assert metrics["cell_count"] == ref["cell_count"]
    assert metrics["edge_count"] == ref["edge_count"]
    assert abs(metrics["avg_neighbors"] - ref["avg_neighbors"]) < 0.10
    assert metrics["hash_vertices_uv"] == ref["hash_vertices_uv"]
    assert metrics["hash_edges"] == ref["hash_edges"]

    assert_png_matches(png_path, png_bytes, mean_abs_tol=1.8)
