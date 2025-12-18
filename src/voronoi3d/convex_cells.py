from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull


@dataclass(frozen=True)
class ConvexPolyhedron:
    """
    Halfspace-defined convex polyhedron.
    halfspaces: (M,4) with rows [a,b,c,d] representing a*x + b*y + c*z + d <= 0
    """
    halfspaces: np.ndarray
    interior_point: np.ndarray  # (3,)

    def vertices(self) -> np.ndarray:
        hs = np.asarray(self.halfspaces, dtype=np.float64)
        ip = np.asarray(self.interior_point, dtype=np.float64)

        # HalfspaceIntersection requires interior point strictly feasible.
        # We'll assume caller ensured that (or nudged).
        hsi = HalfspaceIntersection(hs, ip)
        return np.asarray(hsi.intersections, dtype=np.float64)

    def volume(self) -> float:
        V = self.vertices()
        if len(V) < 4:
            return 0.0
        hull = ConvexHull(V)
        return float(hull.volume)

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        V = self.vertices()
        return V.min(axis=0), V.max(axis=0)

    def contains(self, points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """
        points: (N,3) -> mask (N,)
        """
        P = np.asarray(points, dtype=np.float64)
        hs = np.asarray(self.halfspaces, dtype=np.float64)
        A = hs[:, :3]
        d = hs[:, 3]
        # For each point, check all inequalities
        val = P @ A.T + d[None, :]
        return np.all(val <= eps, axis=1)


def _box_halfspaces(size_xyz: Tuple[float, float, float]) -> np.ndarray:
    Lx, Ly, Lz = map(float, size_xyz)
    # a*x + b*y + c*z + d <= 0
    # x >= 0  => (-1)x + 0 <= 0
    # x <= Lx => ( 1)x - Lx <= 0
    hs = [
        [-1.0, 0.0, 0.0, 0.0],
        [ 1.0, 0.0, 0.0, -Lx],
        [0.0, -1.0, 0.0, 0.0],
        [0.0,  1.0, 0.0, -Ly],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0,  1.0, -Lz],
    ]
    return np.asarray(hs, dtype=np.float64)


def voronoi_cell_halfspaces_in_box(
    seeds_xyz: np.ndarray,
    cell_index: int,
    size_xyz: Tuple[float, float, float],
) -> ConvexPolyhedron:
    """
    Build convex cell polyhedron for seed i within an axis-aligned box,
    using bisector halfspaces vs all other seeds + box boundary halfspaces.
    """
    S = np.asarray(seeds_xyz, dtype=np.float64)
    i = int(cell_index)
    si = S[i]

    hs = [_box_halfspaces(size_xyz)]

    # bisectors: (sj - si)·x <= (sj - si)·m
    for j in range(len(S)):
        if j == i:
            continue
        sj = S[j]
        n = sj - si
        nn = float(np.linalg.norm(n))
        if nn < 1e-12:
            continue
        m = 0.5 * (si + sj)
        # inequality: n·x - n·m <= 0  => [n, -n·m]
        d = -float(np.dot(n, m))
        hs.append(np.array([n[0], n[1], n[2], d], dtype=np.float64)[None, :])

    halfspaces = np.vstack(hs)

    # Ensure interior point is strictly feasible:
    # If a seed lies exactly on a plane numerically, nudge it slightly towards inside of box.
    ip = si.copy()
    eps = 1e-7
    ip[0] = np.clip(ip[0], eps, float(size_xyz[0]) - eps)
    ip[1] = np.clip(ip[1], eps, float(size_xyz[1]) - eps)
    ip[2] = np.clip(ip[2], eps, float(size_xyz[2]) - eps)

    return ConvexPolyhedron(halfspaces=halfspaces, interior_point=ip)


def sample_points_in_convex_polyhedron(
    poly: ConvexPolyhedron,
    n: int,
    rng: np.random.Generator,
    *,
    max_tries: int = 2_000_000,
) -> np.ndarray:
    """
    Rejection-sample inside a convex polyhedron (halfspace form),
    using its vertex bbox as proposal.

    Deterministic given rng seed.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    vmin, vmax = poly.bbox()
    vmin = vmin.astype(np.float64)
    vmax = vmax.astype(np.float64)

    out = []
    tries = 0
    batch = max(256, n * 4)

    while len(out) < n and tries < max_tries:
        k = min(batch, n - len(out))
        P = rng.random((k * 4, 3)) * (vmax - vmin) + vmin
        mask = poly.contains(P)
        accepted = P[mask]
        for p in accepted:
            out.append(p)
            if len(out) >= n:
                break
        tries += len(P)

    if len(out) < n:
        raise RuntimeError(f"Sampling failed: needed {n} points, got {len(out)} (tries={tries}). "
                           f"Try fewer points or check polyhedron feasibility.")

    return np.asarray(out[:n], dtype=np.float64)
