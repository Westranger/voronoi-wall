from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from src.voronoi3d.hierarchy import build_hierarchy_box
from tests.voronoi3d.helpers_golden3d import (
    save_png,
    assert_png_matches,
    save_metrics,
    load_metrics,
    update_mode,
)

GOLDEN_ROOT = Path("goldens") / "voronoi3d_hierarchy"


# -----------------------------
# Polyhedra slice renderer
# -----------------------------
def render_slice_png_polyhedra(
    polyhedra,
    *,
    axis: str = "z",
    value: float | None = None,
    size_px=(1000, 700),
    padding_px=10,
) -> bytes:
    """
    Slice a list of convex polyhedra by plane coord[axis]=value and render intersection segments as PNG.

    This is deterministic, does NOT depend on trimesh boolean or plotting.
    Works directly on your core representation: ConvexPolyhedron.vertices().
    """
    ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    keep_axes = [0, 1, 2]
    keep_axes.remove(ax)
    a2, b2 = keep_axes

    # gather vertices for choosing default slice position
    all_verts = []
    for p in polyhedra:
        V = p.vertices()
        if isinstance(V, np.ndarray) and V.ndim == 2 and V.shape[1] == 3 and len(V) >= 4:
            all_verts.append(V)

    W, H = size_px
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if not all_verts:
        draw.text((padding_px, padding_px), "NO POLYHEDRA", fill=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    VV = np.vstack(all_verts)
    if value is None:
        value = float(0.5 * (VV[:, ax].min() + VV[:, ax].max()))

    segments_2d = []

    def tri_plane_segment(tri: np.ndarray):
        # tri: (3,3), plane: coord[ax]=value
        d = tri[:, ax] - value
        pts = []

        for i in range(3):
            p0 = tri[i]
            p1 = tri[(i + 1) % 3]
            d0 = float(d[i])
            d1 = float(d[(i + 1) % 3])

            if abs(d0) < 1e-9 and abs(d1) < 1e-9:
                pts.append(p0)
                pts.append(p1)
            elif abs(d0) < 1e-9:
                pts.append(p0)
            elif abs(d1) < 1e-9:
                pts.append(p1)
            elif d0 * d1 < 0:
                t = d0 / (d0 - d1)
                pts.append(p0 + t * (p1 - p0))

        if len(pts) < 2:
            return None

        P = np.unique(np.round(np.array(pts), 9), axis=0)
        if len(P) < 2:
            return None

        if len(P) > 2:
            best = None
            bestd = -1.0
            for i in range(len(P)):
                for j in range(i + 1, len(P)):
                    dd = float(np.sum((P[i] - P[j]) ** 2))
                    if dd > bestd:
                        bestd = dd
                        best = (P[i], P[j])
            p, q = best
        else:
            p, q = P[0], P[1]

        return (p, q)

    # accumulate segments
    for p in polyhedra:
        V = p.vertices()
        if not (isinstance(V, np.ndarray) and V.ndim == 2 and V.shape[1] == 3 and len(V) >= 4):
            continue

        hull = ConvexHull(V)
        for tri_idx in hull.simplices:
            tri = V[np.array(tri_idx, dtype=int)]
            seg = tri_plane_segment(tri)
            if seg is None:
                continue
            p3, q3 = seg
            segments_2d.append(((float(p3[a2]), float(p3[b2])), (float(q3[a2]), float(q3[b2]))))

    if len(segments_2d) == 0:
        # red X means: nothing intersects the plane
        draw.line([(padding_px, padding_px), (W - padding_px, H - padding_px)], fill=(255, 0, 0), width=6)
        draw.line([(W - padding_px, padding_px), (padding_px, H - padding_px)], fill=(255, 0, 0), width=6)
        draw.text((padding_px + 5, padding_px + 5), f"NO SEGMENTS axis={axis} value={value:.3f}", fill=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    pts = np.array([p for seg in segments_2d for p in seg], dtype=np.float64)
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    spanx = max(maxx - minx, 1e-9)
    spany = max(maxy - miny, 1e-9)

    sx = (W - 2 * padding_px) / spanx
    sy = (H - 2 * padding_px) / spany
    s = min(sx, sy)

    def map2(pt):
        x = padding_px + (pt[0] - minx) * s
        y = padding_px + (pt[1] - miny) * s
        return (float(x), float(y))

    for (p2, q2) in segments_2d:
        draw.line([map2(p2), map2(q2)], fill=(0, 0, 0), width=2)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# -----------------------------
# Hierarchy extraction helpers
# -----------------------------
def _level0_polyhedra(h):
    return [c.poly for c in h.root_cells]


def _refined_polyhedra(h):
    """
    Return a flat list where refined parents are replaced by their children.
    """
    polys = []
    for c in h.root_cells:
        if getattr(c, "children", None):
            polys.extend([ch.poly for ch in c.children])
        else:
            polys.append(c.poly)
    return polys


def _children_polyhedra(h, parent_index: int):
    c = h.root_cells[int(parent_index)]
    return [ch.poly for ch in c.children]


def compute_hierarchy_metrics(h) -> dict:
    """
    Deterministic regression metrics based on your hierarchy objects.
    """
    root_n = len(h.root_cells)
    refined_parent_n = sum(1 for c in h.root_cells if getattr(c, "children", None))
    child_n = sum(len(c.children) for c in h.root_cells if getattr(c, "children", None))

    polys_ref = _refined_polyhedra(h)
    vols = [float(p.volume()) for p in polys_ref]
    total_vol = float(np.sum(vols)) if vols else 0.0

    return {
        "root_cell_count": int(root_n),
        "refined_parent_count": int(refined_parent_n),
        "child_cell_count": int(child_n),
        "final_cell_count": int(len(polys_ref)),
        "total_volume": float(total_vol),
    }


# -----------------------------
# Tests
# -----------------------------
def test_golden_hierarchy_two_seeds_refine_left():
    case_dir = GOLDEN_ROOT / "HG01_two_seeds_refine_left"
    case_dir.mkdir(parents=True, exist_ok=True)

    png0 = case_dir / "slice_level0.png"
    png1 = case_dir / "slice_refined.png"
    png2 = case_dir / "slice_focus_left_children.png"
    jpath = case_dir / "metrics.json"

    size = (10.0, 10.0, 10.0)
    seeds0 = np.array([[2.5, 5.0, 5.0], [7.5, 5.0, 5.0]], dtype=np.float64)
    rng = np.random.default_rng(123)

    def pred(i: int, seeds: np.ndarray) -> bool:
        return int(i) == 0  # refine left

    h = build_hierarchy_box(
        size_xyz=size,
        seeds_level0=seeds0,
        refine_predicate=pred,
        n_child_seeds=12,
        rng=rng,
    )

    img0 = render_slice_png_polyhedra(_level0_polyhedra(h), axis="z", value=5.0)
    img1 = render_slice_png_polyhedra(_refined_polyhedra(h), axis="z", value=5.0)
    img2 = render_slice_png_polyhedra(_children_polyhedra(h, 0), axis="z", value=5.0)

    metrics = compute_hierarchy_metrics(h)

    if update_mode():
        save_metrics(jpath, metrics)
        save_png(png0, img0)
        save_png(png1, img1)
        save_png(png2, img2)
        return

    ref = load_metrics(jpath)
    assert metrics == ref

    assert_png_matches(png0, img0, mean_abs_tol=2.0)
    assert_png_matches(png1, img1, mean_abs_tol=2.0)
    assert_png_matches(png2, img2, mean_abs_tol=2.0)


def test_golden_hierarchy_four_seeds_refine_diagonal():
    case_dir = GOLDEN_ROOT / "HG02_four_seeds_refine_diagonal"
    case_dir.mkdir(parents=True, exist_ok=True)

    png0 = case_dir / "slice_level0.png"
    png1 = case_dir / "slice_refined.png"
    jpath = case_dir / "metrics.json"

    size = (10.0, 10.0, 10.0)
    seeds0 = np.array(
        [
            [2.5, 2.5, 5.0],
            [7.5, 2.5, 5.0],
            [2.5, 7.5, 5.0],
            [7.5, 7.5, 5.0],
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(999)

    def pred(i: int, seeds: np.ndarray) -> bool:
        return int(i) in (0, 3)  # refine diagonal

    h = build_hierarchy_box(
        size_xyz=size,
        seeds_level0=seeds0,
        refine_predicate=pred,
        n_child_seeds=10,
        rng=rng,
    )

    img0 = render_slice_png_polyhedra(_level0_polyhedra(h), axis="z", value=5.0)
    img1 = render_slice_png_polyhedra(_refined_polyhedra(h), axis="z", value=5.0)

    metrics = compute_hierarchy_metrics(h)

    if update_mode():
        save_metrics(jpath, metrics)
        save_png(png0, img0)
        save_png(png1, img1)
        return

    ref = load_metrics(jpath)
    assert metrics == ref

    assert_png_matches(png0, img0, mean_abs_tol=2.0)
    assert_png_matches(png1, img1, mean_abs_tol=2.0)
