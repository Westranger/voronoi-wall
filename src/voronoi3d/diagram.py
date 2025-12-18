from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Set

import numpy as np
import trimesh


@dataclass(frozen=True)
class VoronoiFace3D:
    """
    A polygonal face (stored by vertex indices into diagram.vertices).
    cell_a is always an original seed cell id.
    cell_b is either another original seed cell id (internal face) or None (exposed face).
    """
    vertex_indices: Tuple[int, ...]
    cell_a: int
    cell_b: Optional[int]  # None => exposed (boundary) face
    is_exposed: bool
    normal: Tuple[float, float, float]


@dataclass
class VoronoiCell3D:
    """
    A cell corresponds to one original seed point.
    """
    seed_index: int
    face_indices: List[int]


@dataclass
class VoronoiDiagram3D:
    """
    Lightweight topological container for a bounded Voronoi in 3D.
    - vertices: (M,3) float64
    - faces: list of polygonal faces (each face is a vertex index loop)
    - cells: list of VoronoiCell3D for each original seed
    - adjacency: dict cell -> set(neighbor cells)
    """
    vertices: np.ndarray
    faces: List[VoronoiFace3D]
    cells: List[VoronoiCell3D]
    seeds: np.ndarray
    adjacency: Dict[int, Set[int]]

    def exposed_face_indices(self) -> List[int]:
        return [i for i, f in enumerate(self.faces) if f.is_exposed]

    def internal_face_indices(self) -> List[int]:
        return [i for i, f in enumerate(self.faces) if (not f.is_exposed) and (f.cell_b is not None)]

    def to_trimesh_surface(self, exposed_only: bool = True) -> trimesh.Trimesh:
        """
        Build a triangle mesh for visualization:
        - exposed_only=True: only outer faces
        - exposed_only=False: all faces (internal + exposed)
        """
        V = self.vertices
        tri_faces = []

        for f in self.faces:
            if exposed_only and not f.is_exposed:
                continue
            vidx = list(f.vertex_indices)
            if len(vidx) < 3:
                continue
            v0 = vidx[0]
            # fan triangulation
            for k in range(1, len(vidx) - 1):
                tri_faces.append([v0, vidx[k], vidx[k + 1]])

        if len(tri_faces) == 0:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)

        m = trimesh.Trimesh(vertices=V.copy(), faces=np.asarray(tri_faces, dtype=int), process=False)
        m.remove_unreferenced_vertices()
        m.process(validate=True)
        return m
