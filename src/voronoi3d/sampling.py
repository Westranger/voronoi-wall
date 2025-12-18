from __future__ import annotations

import numpy as np


def sample_points_in_box(
    size_xyz: tuple[float, float, float],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Uniform sampling in axis-aligned box [0..Lx] x [0..Ly] x [0..Lz]
    """
    Lx, Ly, Lz = map(float, size_xyz)
    if n < 0:
        raise ValueError("n must be >= 0")
    pts = np.empty((n, 3), dtype=np.float64)
    pts[:, 0] = rng.random(n) * Lx
    pts[:, 1] = rng.random(n) * Ly
    pts[:, 2] = rng.random(n) * Lz
    return pts


def sample_points_in_box_by_target_volume(
    size_xyz: tuple[float, float, float],
    target_cell_volume_mm3: float,
    rng: np.random.Generator,
    *,
    min_points: int = 1,
    max_points: int = 20000,
) -> np.ndarray:
    """
    Instead of 'n points', specify target cell volume. We derive n ~ box_volume / target_volume.
    """
    if target_cell_volume_mm3 <= 0:
        raise ValueError("target_cell_volume_mm3 must be > 0")

    Lx, Ly, Lz = map(float, size_xyz)
    box_vol = Lx * Ly * Lz
    n = int(round(box_vol / float(target_cell_volume_mm3)))
    n = max(int(min_points), min(int(max_points), n))
    return sample_points_in_box(size_xyz, n, rng)
