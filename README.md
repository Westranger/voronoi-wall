# VoronoiWall2D – Voronoi 2D on Arbitrary 3D Planes

This repository contains the foundation for a multi-stage, volumetric fracture pipeline.
We start with a robust and testable base module:

- `voronoi2d`: compute a 2D Voronoi partition inside a planar polygon that can be positioned arbitrarily in 3D.
- Supports sampling regions (multiple polygons with different target cell areas or seed counts).
- Returns not only geometry but also topology: cells, edges, neighbors.

## Why 2D first?
Stage 3 (microfracture + crack-like visual behavior) will be driven by surface partitions.
2D Voronoi on surfaces is:
- easy to validate visually,
- deterministic,
- gives clean adjacency graphs.

## Planned modules (later steps)
- `voronoi3d`: voxel-based Voronoi partition inside a closed 3D mesh domain
- selection helpers: exposed patches, adjacency traversal
- pipeline stages: microfracture, chipping, etc.

## Setup
Create a virtualenv and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```
## voronoi3d (Box-only core)

`src/voronoi3d` provides a bounded 3D Voronoi core for axis-aligned boxes using SciPy Voronoi + ghost reflections.

It outputs a topological diagram:
- vertices
- polygonal faces (internal + exposed)
- per-cell face lists
- cell adjacency graph

Run tests:
```powershell
pytest -q
```


