from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple, Dict, Set, List

import numpy as np
from scipy.spatial import Voronoi

from src.voronoi3d.diagram import VoronoiDiagram3D, VoronoiCell3D, VoronoiFace3D


def _newell_normal(poly: np.ndarray) -> np.ndarray:
    """
    Robust polygon normal for possibly non-triangulated polygon.
    """
    n = np.zeros(3, dtype=np.float64)
    for i in range(len(poly)):
        p0 = poly[i]
        p1 = poly[(i + 1) % len(poly)]
        n[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
        n[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
        n[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
    norm = float(np.linalg.norm(n))
    return n if norm < 1e-12 else (n / norm)


def _make_reflections_3d(points: np.ndarray, size_xyz: tuple[float, float, float]) -> np.ndarray:
    """
    Ghost reflections around an axis-aligned box:
    for each axis we use x, -x, 2L-x, giving 27 tiles (minus the original).
    """
    Lx, Ly, Lz = map(float, size_xyz)

    fx_list = [lambda x: x, lambda x: -x, lambda x: 2 * Lx - x]
    fy_list = [lambda y: y, lambda y: -y, lambda y: 2 * Ly - y]
    fz_list = [lambda z: z, lambda z: -z, lambda z: 2 * Lz - z]

    ghosts = []
    for fx in fx_list:
        for fy in fy_list:
            for fz in fz_list:
                if fx is fx_list[0] and fy is fy_list[0] and fz is fz_list[0]:
                    continue
                g = np.column_stack([
                    fx(points[:, 0]),
                    fy(points[:, 1]),
                    fz(points[:, 2]),
                ])
                ghosts.append(g)

    return np.vstack(ghosts) if ghosts else np.zeros((0, 3), dtype=np.float64)


def compute_voronoi_3d(
        *,
        size_xyz: tuple[float, float, float],
        seeds_xyz: Optional[np.ndarray] = None,
        n_points: Optional[int] = None,
        target_cell_volume_mm3: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        bounded: bool = True,
        reflection_diagonals: bool = True,  # reserved for future variants; kept for parity
        weld_decimals: int = 6,
) -> VoronoiDiagram3D:
    """
    Compute a bounded 3D Voronoi for an axis-aligned box volume [0..Lx]x[0..Ly]x[0..Lz].

    For now:
    - Box-only (convex/axis-aligned)
    - Uses SciPy Voronoi + ghost reflections to avoid infinite regions
    - Builds a *face graph* (polygonal faces), cells and adjacency.

    Notes:
    - This does NOT output closed per-cell polyhedra yet.
    - It *does* provide enough topology for later pipeline steps.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    Lx, Ly, Lz = map(float, size_xyz)
    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise ValueError("size_xyz must be positive in all components")

    if seeds_xyz is None:
        if n_points is None and target_cell_volume_mm3 is None:
            raise ValueError("Provide seeds_xyz OR (n_points OR target_cell_volume_mm3)")
        if n_points is None:
            # derive n from target volume
            n_points = int(round((Lx * Ly * Lz) / float(target_cell_volume_mm3)))
            n_points = max(1, n_points)
        seeds_xyz = np.empty((int(n_points), 3), dtype=np.float64)
        seeds_xyz[:, 0] = rng.random(int(n_points)) * Lx
        seeds_xyz[:, 1] = rng.random(int(n_points)) * Ly
        seeds_xyz[:, 2] = rng.random(int(n_points)) * Lz
    else:
        seeds_xyz = np.asarray(seeds_xyz, dtype=np.float64)
        if seeds_xyz.ndim != 2 or seeds_xyz.shape[1] != 3:
            raise ValueError("seeds_xyz must be (N,3)")

    n_orig = int(len(seeds_xyz))

    if bounded:
        ghosts = _make_reflections_3d(seeds_xyz, size_xyz)
        all_points = np.vstack([seeds_xyz, ghosts]) if len(ghosts) else seeds_xyz
    else:
        all_points = seeds_xyz

    vor = Voronoi(all_points)

    # weld global vertices (Voronoi vertices -> our vertex list)
    vertices: List[List[float]] = []
    vmap: Dict[Tuple[float, float, float], int] = {}

    def get_vid(p3: np.ndarray) -> int:
        key = (
            round(float(p3[0]), weld_decimals),
            round(float(p3[1]), weld_decimals),
            round(float(p3[2]), weld_decimals),
        )
        if key not in vmap:
            vmap[key] = len(vertices)
            vertices.append([key[0], key[1], key[2]])
        return vmap[key]

    faces: List[VoronoiFace3D] = []
    cell_faces: List[List[int]] = [[] for _ in range(n_orig)]
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_orig)}

    # Build polygon faces from ridge polygons
    for (i, j), ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge_verts or len(ridge_verts) < 3:
            continue

        i_is_orig = i < n_orig
        j_is_orig = j < n_orig

        # Only build faces relevant to original cells
        if not (i_is_orig or j_is_orig):
            continue

        poly = vor.vertices[np.asarray(ridge_verts, dtype=int)]
        if poly.shape[0] < 3:
            continue

        # Decide which original cell "owns" the face record orientation
        if i_is_orig:
            cell_a = int(i)
            other = int(j)
        else:
            cell_a = int(j)
            other = int(i)

        # Determine if exposed or internal:
        is_internal = (i_is_orig and j_is_orig)
        cell_b = int(other) if is_internal else None
        is_exposed = not is_internal

        # Orient polygon normal to point from cell_a towards other seed/ghost
        outward_dir = all_points[other] - all_points[cell_a]
        nd = float(np.linalg.norm(outward_dir))
        if nd < 1e-12:
            continue
        outward_dir /= nd

        n = _newell_normal(poly)
        if float(np.dot(n, outward_dir)) < 0:
            poly = poly[::-1]
            n = -n

        vidx = tuple(get_vid(p) for p in poly)
        if len(vidx) < 3:
            continue

        face = VoronoiFace3D(
            vertex_indices=vidx,
            cell_a=cell_a,
            cell_b=cell_b,
            is_exposed=is_exposed,
            normal=(float(n[0]), float(n[1]), float(n[2])),
        )
        f_idx = len(faces)
        faces.append(face)

        # Attach to cell_a
        if cell_a < n_orig:
            cell_faces[cell_a].append(f_idx)

        # Attach to cell_b too (internal)
        if cell_b is not None and cell_b < n_orig:
            cell_faces[cell_b].append(f_idx)
            adjacency[cell_a].add(cell_b)
            adjacency[cell_b].add(cell_a)

    V = np.asarray(vertices, dtype=np.float64)
    cells = [VoronoiCell3D(seed_index=i, face_indices=cell_faces[i]) for i in range(n_orig)]

    return VoronoiDiagram3D(
        vertices=V,
        faces=faces,
        cells=cells,
        seeds=seeds_xyz.copy(),
        adjacency=adjacency,
    )
