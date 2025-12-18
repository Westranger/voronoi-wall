from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image


GOLDEN_ROOT = Path("goldens") / "voronoi2d"


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def hash_array_quantized(arr: np.ndarray, decimals: int = 4) -> str:
    """
    Quantize + sort rows + hash.
    Robust against minor float noise and row ordering.
    """
    if arr.size == 0:
        return _sha256_bytes(b"empty")

    q = np.round(arr.astype(np.float64), decimals=decimals)
    if q.ndim == 1:
        q = q.reshape(-1, 1)

    # sort rows
    q2 = q[np.lexsort(q.T[::-1])]
    return _sha256_bytes(q2.tobytes())


def hash_edges(edges) -> str:
    """
    Hash edges as sorted tuples.
    """
    tuples = []
    for e in edges:
        # edge: v0,v1,cells tuple
        a = int(min(e.v0, e.v1))
        b = int(max(e.v0, e.v1))
        cs = tuple(int(x) for x in e.cells)
        tuples.append((a, b, cs))
    tuples.sort()
    raw = repr(tuples).encode("utf-8")
    return _sha256_bytes(raw)


def compute_metrics(diagram) -> dict:
    """
    Numeric metrics used in golden tests.
    """
    cell_count = len(diagram.cells)
    edge_count = len(diagram.edges)
    avg_neighbors = float(np.mean([len(c.neighbors) for c in diagram.cells])) if cell_count else 0.0

    bounds_uv = None
    if len(diagram.vertices_uv):
        mn = diagram.vertices_uv.min(axis=0).tolist()
        mx = diagram.vertices_uv.max(axis=0).tolist()
        bounds_uv = {"min": mn, "max": mx}

    metrics = {
        "cell_count": cell_count,
        "edge_count": edge_count,
        "avg_neighbors": avg_neighbors,
        "hash_vertices_uv": hash_array_quantized(diagram.vertices_uv, decimals=4),
        "hash_edges": hash_edges(diagram.edges),
        "bounds_uv": bounds_uv,
    }
    return metrics


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_png(path: Path, png_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


def png_mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    a = a.convert("RGBA")
    b = b.convert("RGBA")
    if a.size != b.size:
        raise AssertionError(f"PNG size differs: {a.size} vs {b.size}")

    arr_a = np.asarray(a, dtype=np.int16)
    arr_b = np.asarray(b, dtype=np.int16)
    diff = np.abs(arr_a - arr_b)
    return float(diff.mean())


def assert_png_matches(golden_path: Path, current_png_bytes: bytes, *, mean_abs_tol: float) -> None:
    golden_img = Image.open(golden_path)
    cur_img = Image.open(Path(os.devnull))  # dummy to satisfy type checkers
    # Open bytes properly:
    from io import BytesIO
    cur_img = Image.open(BytesIO(current_png_bytes))

    mad = png_mean_abs_diff(golden_img, cur_img)
    if mad > mean_abs_tol:
        raise AssertionError(f"PNG mean abs diff too high: {mad:.4f} > {mean_abs_tol}")


def update_mode() -> bool:
    return os.environ.get("UPDATE_GOLDENS", "0") == "1"
