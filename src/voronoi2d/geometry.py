import numpy as np


def plane_from_polygon(polygon_xyz: np.ndarray):
    """
    Compute plane origin + orthonormal basis (u, v, n)
    polygon_xyz: (N,3), assumed non-degenerate
    """
    p0 = polygon_xyz[0]
    v1 = polygon_xyz[1] - p0
    v2 = polygon_xyz[2] - p0
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)

    u = v1 / np.linalg.norm(v1)
    v = np.cross(n, u)

    return p0, u, v, n


def project_xyz_to_uv(points_xyz: np.ndarray, origin, u, v):
    d = points_xyz - origin
    return np.column_stack([d @ u, d @ v])


def lift_uv_to_xyz(points_uv: np.ndarray, origin, u, v):
    return origin + np.outer(points_uv[:, 0], u) + np.outer(points_uv[:, 1], v)
