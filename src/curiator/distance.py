"""ROI-weighted distances between volume-level Curia embeddings."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import SliceEncoder, rescale_volume


def pool_depth(x: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Average-pool ``x`` of shape ``[D, ...]`` along D with kernel = stride = ``pool_size``.

    A trailing partial window is averaged over the slices it contains.
    """
    if pool_size <= 1:
        return x
    shape = x.shape
    flat = x.reshape(shape[0], -1).t().unsqueeze(0)  # [1, F, D]
    pooled = F.avg_pool1d(flat, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    return pooled.squeeze(0).t().reshape(-1, *shape[1:])


def token_weights(roi_crop: np.ndarray, grid_size: int, pool_size: int = 8) -> torch.Tensor:
    """Project a cropped binary ROI ``[D, H, W]`` onto the token grid.

    Returns fractional overlap weights ``w`` in [0, 1] of shape ``[D_r, P]``.
    """
    m = torch.from_numpy(roi_crop.astype(np.float32)).unsqueeze(1)  # [D, 1, H, W]
    w = F.interpolate(m, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    return pool_depth(w.reshape(roi_crop.shape[0], -1), pool_size)


def embed_volume(volume_crop: np.ndarray, encoder: SliceEncoder, pool_size: int = 8) -> torch.Tensor:
    """Embed a cropped volume ``[D, H, W]`` slice by slice.

    Returns the depth-pooled volume embedding ``E`` of shape ``[D_r, P, C]``.
    """
    vol = rescale_volume(volume_crop.astype(np.float32))
    tokens = encoder([vol[d] for d in range(vol.shape[0])])  # [D, P, C]
    return pool_depth(tokens, pool_size)


def volume_distance(e_i: torch.Tensor, e_j: torch.Tensor, weights: torch.Tensor) -> float:
    """Masked, spatially weighted distance between two volume embeddings.

    ``D_ij = mean_d sqrt( sum_p w_dp * delta_dp / sum_p w_dp )`` with
    ``delta_dp = mean_c (E^i_dpc - E^j_dpc)^2``.
    """
    delta = (e_i - e_j).pow(2).mean(dim=-1)  # [D_r, P]
    w_sum = weights.sum(dim=-1).clamp_min(1e-6)
    per_slice = ((delta * weights).sum(dim=-1) / w_sum).clamp_min(0.0).sqrt()
    return float(per_slice.mean())


def distance_matrix(embeddings: Sequence[torch.Tensor], weights: torch.Tensor) -> np.ndarray:
    """Symmetric ``[N, N]`` matrix of :func:`volume_distance` over all embedding pairs."""
    n = len(embeddings)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = volume_distance(embeddings[i], embeddings[j], weights)
    return dist
