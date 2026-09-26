"""Distance-matrix visualisation (requires the ``plot`` extra)."""

from __future__ import annotations

from os import PathLike
from typing import Optional, Sequence, Union

import numpy as np


def plot_distance_matrix(
    dist: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    selected: Optional[int] = None,
    out: Optional[Union[str, PathLike]] = None,
    title: Optional[str] = None,
):
    """Plot a CURIAtor distance matrix; the selected phase's row/column is outlined."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = dist.shape[0]
    labels = list(labels) if labels else ["Pre"] + [f"Post{i}" for i in range(1, n)]
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * n, 0.6 + 0.7 * n))
    im = ax.imshow(dist, cmap="viridis", vmin=0.0, vmax=max(1.0, float(dist.max())))
    ax.set_xticks(range(n), labels, rotation=45, ha="left")
    ax.set_yticks(range(n), labels)
    ax.xaxis.tick_top()
    if selected is not None:
        # Outline D[selected, Post1] and its mirror entry.
        for col, row in [(1, selected), (selected, 1)]:
            ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, ec="white", lw=2.5))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title, pad=40)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig
