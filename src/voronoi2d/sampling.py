import numpy as np
from shapely.geometry import Polygon, Point


def sample_points_in_polygon(
    polygon_uv: np.ndarray,
    *,
    target_area_mm2: float | None = None,
    n_points: int | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    poly = Polygon(polygon_uv)
    area = poly.area

    if n_points is None:
        if target_area_mm2 is None:
            raise ValueError("Either target_area_mm2 or n_points required")
        n_points = max(1, int(area / target_area_mm2))

    minx, miny, maxx, maxy = poly.bounds

    points = []
    while len(points) < n_points:
        p = Point(
            rng.uniform(minx, maxx),
            rng.uniform(miny, maxy),
        )
        if poly.contains(p):
            points.append((p.x, p.y))

    return np.array(points, dtype=np.float64)
