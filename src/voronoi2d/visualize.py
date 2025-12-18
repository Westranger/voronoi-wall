import matplotlib.pyplot as plt


def plot_voronoi_uv(diagram, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for cell in diagram.cells:
        p = cell.polygon_uv
        ax.plot(*p.T, "-k")

    ax.set_aspect("equal")
    ax.set_title("Voronoi2D (UV)")
    return ax
