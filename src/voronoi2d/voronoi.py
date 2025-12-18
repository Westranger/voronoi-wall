import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from collections import defaultdict

from .datastructures import VoronoiDiagram2D, VoronoiCell2D, VoronoiEdge2D
from .geometry import lift_uv_to_xyz


def _make_reflections_2d(points_uv: np.ndarray, bounds, include_diagonals: bool = True) -> np.ndarray:
    """
    Create mirrored ghost points around a bounding box to make Voronoi regions finite.

    bounds: (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = map(float, bounds)

    # reflect x and y around min/max
    fx = [
        lambda x: x,
        lambda x: 2 * minx - x,
        lambda x: 2 * maxx - x,
    ]
    fy = [
        lambda y: y,
        lambda y: 2 * miny - y,
        lambda y: 2 * maxy - y,
    ]

    ghosts = []
    for ix, fxi in enumerate(fx):
        for iy, fyi in enumerate(fy):
            if ix == 0 and iy == 0:
                continue
            if not include_diagonals:
                # only axis reflections (skip diagonal combinations)
                if ix != 0 and iy != 0:
                    continue

            g = np.column_stack([
                fxi(points_uv[:, 0]),
                fyi(points_uv[:, 1])
            ])
            ghosts.append(g)

    return np.vstack(ghosts) if ghosts else np.zeros((0, 2), dtype=np.float64)


def compute_voronoi_2d(
    polygon_uv: np.ndarray,
    polygon_xyz: np.ndarray,
    seeds_uv: np.ndarray,
    origin, u, v,
    *,
    bounded: bool = True,
    reflection_diagonals: bool = True,
    weld_decimals: int = 6,
) -> VoronoiDiagram2D:
    """
    Compute bounded 2D Voronoi diagram inside polygon_uv, mapped back to 3D plane via origin,u,v.

    Key design detail:
    - SciPy Voronoi produces infinite regions for hull points.
    - For bounded polygons, we stabilize by adding mirrored ghost points around polygon bounds.
    - We only return cells for the original seeds.
    """
    poly = Polygon(polygon_uv)
    if poly.is_empty or not poly.is_valid:
        raise ValueError("Input polygon_uv must be a valid, non-empty polygon")

    seeds_uv = np.asarray(seeds_uv, dtype=np.float64)
    if seeds_uv.ndim != 2 or seeds_uv.shape[1] != 2:
        raise ValueError("seeds_uv must be (N,2)")

    n_orig = len(seeds_uv)

    # Add reflections to make regions finite
    if bounded:
        ghosts = _make_reflections_2d(seeds_uv, poly.bounds, include_diagonals=reflection_diagonals)
        all_seeds = np.vstack([seeds_uv, ghosts]) if len(ghosts) else seeds_uv
    else:
        all_seeds = seeds_uv

    vor = Voronoi(all_seeds)

    vertices = []
    vertex_map = {}

    def get_vertex_index(pt2):
        key = (round(float(pt2[0]), weld_decimals), round(float(pt2[1]), weld_decimals))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append([key[0], key[1]])
        return vertex_map[key]

    cells = []
    edge_map = defaultdict(set)  # (va,vb) -> {cell_ids...}

    # iterate only original points
    for i in range(n_orig):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        # with reflections, regions should be finite; still guard:
        if -1 in region or len(region) < 3:
            continue

        poly_cell = Polygon(vor.vertices[region])
        clipped = poly_cell.intersection(poly)

        if clipped.is_empty:
            continue

        # handle MultiPolygon by taking largest part
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)

        if clipped.is_empty or clipped.geom_type != "Polygon":
            continue

        coords = np.array(clipped.exterior.coords[:-1], dtype=np.float64)
        if len(coords) < 3:
            continue

        vidx = [get_vertex_index(p) for p in coords]

        # edges for topology
        for a, b in zip(vidx, vidx[1:] + [vidx[0]]):
            edge_map[tuple(sorted((a, b)))].add(i)

        cells.append(
            VoronoiCell2D(
                index=i,
                polygon_uv=coords,
                polygon_xyz=lift_uv_to_xyz(coords, origin, u, v),
                neighbors=[],
            )
        )

    # Build edges list
    edges = []
    for (a, b), cs in edge_map.items():
        edges.append(VoronoiEdge2D(a, b, tuple(sorted(cs))))

    # Build neighbor graph
    neighbors = defaultdict(set)
    for e in edges:
        if len(e.cells) == 2:
            c0, c1 = e.cells
            neighbors[c0].add(c1)
            neighbors[c1].add(c0)

    # attach to cells
    cell_by_id = {c.index: c for c in cells}
    for cid, nbs in neighbors.items():
        if cid in cell_by_id:
            cell_by_id[cid].neighbors = sorted(nbs)

    vertices_uv = np.array(vertices, dtype=np.float64) if vertices else np.zeros((0, 2), dtype=np.float64)
    vertices_xyz = lift_uv_to_xyz(vertices_uv, origin, u, v) if len(vertices_uv) else np.zeros((0, 3), dtype=np.float64)

    return VoronoiDiagram2D(
        vertices_uv=vertices_uv,
        vertices_xyz=vertices_xyz,
        cells=cells,
        edges=edges,
    )
