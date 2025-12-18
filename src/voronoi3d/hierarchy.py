from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .convex_cells import (
    ConvexPolyhedron,
    voronoi_cell_halfspaces_in_box,
    sample_points_in_convex_polyhedron,
)


@dataclass
class HierCell:
    """
    One node in the hierarchy: a convex region + its seed id.
    For parent level, seed_index refers to the global seed index of that level's seed array.
    """
    poly: ConvexPolyhedron
    seed_index: int
    children: List["HierCell"]


@dataclass
class Hierarchy3D:
    """
    Root-level seeds in box + optional refinement per cell.
    """
    size_xyz: Tuple[float, float, float]
    seeds_level0: np.ndarray
    root_cells: List[HierCell]


def build_level0_cells_box(
    *,
    size_xyz: Tuple[float, float, float],
    seeds_xyz: np.ndarray,
) -> List[HierCell]:
    """
    Compute convex polyhedron for every seed in the box using halfspace intersection.
    """
    S = np.asarray(seeds_xyz, dtype=np.float64)
    cells: List[HierCell] = []
    for i in range(len(S)):
        poly = voronoi_cell_halfspaces_in_box(S, i, size_xyz)
        cells.append(HierCell(poly=poly, seed_index=i, children=[]))
    return cells


def refine_cell_in_box(
    *,
    size_xyz: Tuple[float, float, float],
    parent_seeds_xyz: np.ndarray,
    parent_cell_index: int,
    n_child_seeds: int,
    rng: np.random.Generator,
) -> List[HierCell]:
    """
    Refine a single level-0 cell by creating child seeds inside its convex polyhedron,
    then computing child Voronoi cells bounded by the SAME parent polyhedron.
    (i.e., bisectors between child seeds + parent halfspaces)
    """
    parent_poly = voronoi_cell_halfspaces_in_box(parent_seeds_xyz, parent_cell_index, size_xyz)

    child_seeds = sample_points_in_convex_polyhedron(parent_poly, int(n_child_seeds), rng)

    # For each child seed, build halfspaces:
    # - all parent halfspaces (bounding)
    # - plus bisectors vs other child seeds
    child_cells: List[HierCell] = []
    S = child_seeds

    for i in range(len(S)):
        si = S[i]
        hs = [np.asarray(parent_poly.halfspaces, dtype=np.float64)]

        for j in range(len(S)):
            if j == i:
                continue
            sj = S[j]
            n = sj - si
            nn = float(np.linalg.norm(n))
            if nn < 1e-12:
                continue
            m = 0.5 * (si + sj)
            d = -float(np.dot(n, m))
            hs.append(np.array([n[0], n[1], n[2], d], dtype=np.float64)[None, :])

        halfspaces = np.vstack(hs)

        # interior point: child seed itself; nudge tiny to remain feasible
        ip = si.copy()
        child_poly = ConvexPolyhedron(halfspaces=halfspaces, interior_point=ip)
        child_cells.append(HierCell(poly=child_poly, seed_index=i, children=[]))

    return child_cells


def build_hierarchy_box(
    *,
    size_xyz: Tuple[float, float, float],
    seeds_level0: np.ndarray,
    refine_predicate: Callable[[int, np.ndarray], bool],
    n_child_seeds: int,
    rng: np.random.Generator,
) -> Hierarchy3D:
    """
    Build hierarchy:
    - level0: Voronoi cells in box for seeds_level0
    - refine: for each level0 cell where predicate(cell_index, seeds_level0) is True,
              refine it with n_child_seeds child seeds (inside the parent poly)
    """
    roots = build_level0_cells_box(size_xyz=size_xyz, seeds_xyz=seeds_level0)

    for cell in roots:
        if refine_predicate(cell.seed_index, seeds_level0):
            cell.children = refine_cell_in_box(
                size_xyz=size_xyz,
                parent_seeds_xyz=seeds_level0,
                parent_cell_index=cell.seed_index,
                n_child_seeds=int(n_child_seeds),
                rng=rng,
            )

    return Hierarchy3D(
        size_xyz=size_xyz,
        seeds_level0=np.asarray(seeds_level0, dtype=np.float64),
        root_cells=roots,
    )
