import numpy as np

from src.voronoi3d.hierarchy import build_hierarchy_box


def test_refine_one_of_two_cells_volume_conservation():
    size = (10.0, 10.0, 10.0)
    seeds0 = np.array([
        [2.5, 5.0, 5.0],
        [7.5, 5.0, 5.0],
    ], dtype=np.float64)

    # refine only cell 0
    def pred(cell_index: int, seeds: np.ndarray) -> bool:
        return int(cell_index) == 0

    rng = np.random.default_rng(123)

    h = build_hierarchy_box(
        size_xyz=size,
        seeds_level0=seeds0,
        refine_predicate=pred,
        n_child_seeds=10,
        rng=rng,
    )

    assert len(h.root_cells) == 2
    c0 = h.root_cells[0]
    c1 = h.root_cells[1]

    assert len(c0.children) == 10
    assert len(c1.children) == 0

    v_parent = c0.poly.volume()
    v_children = sum(ch.poly.volume() for ch in c0.children)

    assert v_parent > 0
    assert v_children > 0

    # child cells partition the parent polyhedron (within numeric tolerance)
    assert abs(v_children - v_parent) / v_parent < 0.05
