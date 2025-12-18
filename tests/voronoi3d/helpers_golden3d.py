from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

from src.voronoi3d.diagram import VoronoiDiagram3D, VoronoiFace3D

GOLDEN_ROOT = Path("goldens") / "voronoi3d"


def update_mode() -> bool:
    v = os.environ.get("UPDATE_GOLDENS", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def save_png(path: Path, png_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


def save_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_array(a: np.ndarray, decimals: int = 6) -> str:
    """
    Stable hash for numeric arrays: round + bytes hash.
    """
    b = np.round(a.astype(np.float64), decimals=decimals)
    h = hashlib.sha256(b.tobytes()).hexdigest()
    return h


def _hash_edges(edges: np.ndarray) -> str:
    """
    edges: (M,2) int
    """
    e = np.asarray(edges, dtype=np.int64)
    e = np.sort(e, axis=1)
    e = e[np.lexsort((e[:, 1], e[:, 0]))]
    h = hashlib.sha256(e.tobytes()).hexdigest()
    return h


def compute_metrics_3d(d: VoronoiDiagram3D) -> Dict[str, Any]:
    """
    Return small, stable summary metrics + hashes.
    """
    # adjacency edges
    adj_pairs = []
    for a, nbs in d.adjacency.items():
        for b in nbs:
            if a < b:
                adj_pairs.append([a, b])
    adj_pairs = np.asarray(adj_pairs, dtype=np.int64) if len(adj_pairs) else np.zeros((0, 2), dtype=np.int64)

    exposed = 0
    internal = 0
    for f in d.faces:
        if f.is_exposed:
            exposed += 1
        else:
            internal += 1

    avg_neighbors = float(np.mean([len(v) for v in d.adjacency.values()])) if len(d.adjacency) else 0.0

    # hash vertices + faces as (cell_a, cell_b, vertex_indices...)
    Vhash = _hash_array(d.vertices, decimals=6)

    # Face hash: build deterministic int representation
    face_rows = []
    for f in d.faces:
        cb = -1 if f.cell_b is None else int(f.cell_b)
        row = [int(f.cell_a), cb] + list(map(int, f.vertex_indices))
        face_rows.append(row)

    # pad to rectangular by hashing list-of-lists as json
    face_json = json.dumps(face_rows, separators=(",", ":"), sort_keys=False).encode("utf-8")
    Fhash = hashlib.sha256(face_json).hexdigest()

    metrics = {
        "cell_count": int(len(d.cells)),
        "face_count": int(len(d.faces)),
        "exposed_face_count": int(exposed),
        "internal_face_count": int(internal),
        "avg_neighbors": float(avg_neighbors),
        "hash_vertices": Vhash,
        "hash_faces": Fhash,
        "hash_adjacency_edges": _hash_edges(adj_pairs),
    }
    return metrics


def assert_png_matches(path: Path, png_bytes: bytes, mean_abs_tol: float) -> None:
    """
    Compare current png_bytes to reference file at path by mean absolute pixel diff.
    """
    ref_bytes = path.read_bytes()
    ref_img = Image.open(BytesIO(ref_bytes)).convert("RGB")
    cur_img = Image.open(BytesIO(png_bytes)).convert("RGB")

    ref = np.asarray(ref_img, dtype=np.int16)
    cur = np.asarray(cur_img, dtype=np.int16)

    assert ref.shape == cur.shape, f"PNG shape mismatch: ref={ref.shape} cur={cur.shape}"

    mean_abs = float(np.mean(np.abs(ref - cur)))
    assert mean_abs <= float(mean_abs_tol), f"PNG mismatch mean_abs={mean_abs:.3f} > tol={mean_abs_tol}"


def _segment_plane_intersection(p0: np.ndarray, p1: np.ndarray, axis: int, value: float) -> Optional[np.ndarray]:
    """
    Intersect segment p0->p1 with plane coord[axis] = value.
    Returns intersection point (3,) or None.
    """
    a0 = float(p0[axis])
    a1 = float(p1[axis])
    if (a0 < value and a1 < value) or (a0 > value and a1 > value):
        return None
    if abs(a1 - a0) < 1e-12:
        return None
    t = (value - a0) / (a1 - a0)
    if t < 0.0 or t > 1.0:
        return None
    return p0 + t * (p1 - p0)


def render_slice_png(
    d: VoronoiDiagram3D,
    size_xyz: Tuple[float, float, float],
    *,
    axis: str = "z",
    value: Optional[float] = None,
    img_size: Tuple[int, int] = (900, 600),
    padding_px: int = 10,
    draw_exposed_only: bool = True,
) -> bytes:
    """
    Render a 2D slice preview by intersecting polygon faces with a plane.
    - axis: "x"|"y"|"z"
    - value: slice position; default mid-plane
    - draw_exposed_only: True => only boundary faces, gives 'wall surface' look
    """
    ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    Lx, Ly, Lz = map(float, size_xyz)
    if value is None:
        value = [Lx * 0.5, Ly * 0.5, Lz * 0.5][ax]

    # project to 2D axes (the other two coords)
    other = [0, 1, 2]
    other.remove(ax)
    a2, b2 = other[0], other[1]

    segments_2d = []

    V = d.vertices
    for f in d.faces:
        if draw_exposed_only and (not f.is_exposed):
            continue

        vidx = list(f.vertex_indices)
        if len(vidx) < 3:
            continue
        pts = V[np.asarray(vidx, dtype=int)]

        # intersect polygon edges with plane, collect intersection points
        hits = []
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            ip = _segment_plane_intersection(p0, p1, ax, float(value))
            if ip is not None:
                hits.append(ip)

        # For a convex-ish polygon slice, we expect 0 or 2 intersections.
        # Some numerical cases produce >2; we take pairs by sorting.
        if len(hits) < 2:
            continue

        # reduce duplicates
        H = np.asarray(hits, dtype=np.float64)
        # sort by a2 then b2
        order = np.lexsort((H[:, b2], H[:, a2]))
        H = H[order]

        # pair neighbors (0-1, 2-3, ...)
        for k in range(0, len(H) - 1, 2):
            p = H[k]
            q = H[k + 1]
            segments_2d.append(((float(p[a2]), float(p[b2])), (float(q[a2]), float(q[b2]))))

    # if nothing, still produce blank image (stable)
    W, Hh = img_size
    img = Image.new("RGB", (W, Hh), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # bounds for mapping
    # Use box bounds for stable framing
    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([Lx, Ly, Lz])
    min2 = mins[[a2, b2]]
    max2 = maxs[[a2, b2]]

    span = max2 - min2
    span[span < 1e-9] = 1.0

    sx = (W - 2 * padding_px) / span[0]
    sy = (Hh - 2 * padding_px) / span[1]
    s = min(sx, sy)

    def map_pt(pt):
        x = padding_px + (pt[0] - float(min2[0])) * s
        y = padding_px + (pt[1] - float(min2[1])) * s
        # image y down; keep y down for stable look
        return (x, y)

    # draw segments
    for (p, q) in segments_2d:
        p2 = map_pt(p)
        q2 = map_pt(q)
        draw.line([p2, q2], fill=(0, 0, 0), width=2)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
