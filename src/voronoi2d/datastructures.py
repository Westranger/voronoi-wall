from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Dict


@dataclass(frozen=True)
class VoronoiEdge2D:
    v0: int
    v1: int
    cells: Tuple[int, ...]  # 1 or 2 cell indices


@dataclass
class VoronoiCell2D:
    index: int
    polygon_uv: np.ndarray  # (N,2)
    polygon_xyz: np.ndarray  # (N,3)
    neighbors: List[int]


@dataclass
class VoronoiDiagram2D:
    vertices_uv: np.ndarray        # (M,2)
    vertices_xyz: np.ndarray       # (M,3)
    cells: List[VoronoiCell2D]
    edges: List[VoronoiEdge2D]

    def cell_count(self) -> int:
        return len(self.cells)

    def edge_count(self) -> int:
        return len(self.edges)
