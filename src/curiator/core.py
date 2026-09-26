"""The CURIAtor phase-selection module."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from os import PathLike
from typing import Optional, Sequence, Union

import numpy as np
import torch

from .distance import distance_matrix, embed_volume, token_weights
from .encoder import CuriaEncoder, SliceEncoder
from .io import BBox, PathOrArray, as_array, bounding_box, crop, roi_from_mask

PRE, POST1 = 0, 1


def select_phase(dist: np.ndarray, anchor: int = POST1, candidates: Optional[Sequence[int]] = None) -> int:
    """Index of the candidate phase most distant from ``anchor`` in ``dist``.

    By default the candidates are all phases after the first post-contrast
    phase (Post1).
    """
    if candidates is None:
        candidates = range(anchor + 1, dist.shape[0])
    candidates = list(candidates)
    if not candidates:
        raise ValueError("No candidate phases to select from.")
    return max(candidates, key=lambda n: dist[n, anchor])


@dataclass
class CURIAtorResult:
    """Outcome of :meth:`CURIAtor.select` for one breast / ROI."""

    distance_matrix: np.ndarray
    selected_index: int
    bbox: BBox
    side: Union[str, int]
    phase_names: list[str] = field(default_factory=list)
    fallback: bool = False  # True if only Pre + Post1 existed and Post1 was reused

    @property
    def triplet_indices(self) -> tuple[int, int, int]:
        """Indices of ``(Pre, Post1, selected)`` into the input phase list."""
        return PRE, POST1, self.selected_index

    @property
    def selected_name(self) -> str:
        return self.phase_names[self.selected_index]

    @property
    def distances_to_post1(self) -> np.ndarray:
        return self.distance_matrix[:, POST1]

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "phases": self.phase_names,
            "selected_index": self.selected_index,
            "selected_phase": self.selected_name,
            "triplet": [self.phase_names[i] for i in self.triplet_indices],
            "fallback": self.fallback,
            "bbox_zyx": list(self.bbox),
            "distance_matrix": self.distance_matrix.round(6).tolist(),
        }


class CURIAtor:
    """Adaptive temporal phase selection for DCE-MRI.

    Given the phases ``[Pre, Post1, Post2, ..., PostN]`` of one acquisition and a
    breast mask, every phase is embedded with the frozen Curia foundation model
    inside the ROI bounding box, a pairwise ROI-weighted distance matrix is
    built, and the post-contrast phase most dissimilar to Post1 is selected.
    The output triplet ``(Pre, Post1, selected)`` can be exported for a classifier.

    Args:
        encoder: Slice encoder; defaults to :class:`CuriaEncoder` (loaded lazily).
        pool_size: Depth average-pooling kernel for token aggregation (default: 8).
        device: Device for the default encoder.
        batch_size: Slices per forward pass of the default encoder.
    """

    def __init__(
        self,
        encoder: Optional[SliceEncoder] = None,
        pool_size: int = 8,
        device: Optional[str | torch.device] = None,
        batch_size: int = 32,
    ):
        self._encoder = encoder
        self.pool_size = pool_size
        self._device = device
        self._batch_size = batch_size

    @property
    def encoder(self) -> SliceEncoder:
        if self._encoder is None:
            self._encoder = CuriaEncoder(device=self._device, batch_size=self._batch_size)
        return self._encoder

    def distance_matrix(
        self,
        phases: Sequence[PathOrArray],
        mask: PathOrArray,
        side: Union[str, int] = "roi",
    ) -> tuple[np.ndarray, BBox]:
        """Pairwise ``[N, N]`` distance matrix of ``phases`` within the ROI, plus the ROI bbox."""
        roi = roi_from_mask(as_array(mask), side)
        bbox = bounding_box(roi)
        if bbox is None:
            raise ValueError(f"ROI {side!r} is empty in the mask.")
        roi_crop = crop(roi, bbox)
        weights = token_weights(roi_crop, self.encoder.grid_size, self.pool_size)

        embeddings = []
        for i, phase in enumerate(phases):
            vol = as_array(phase)
            if vol.shape != roi.shape:
                raise ValueError(f"Phase {i} has shape {vol.shape}, mask has shape {roi.shape}.")
            embeddings.append(embed_volume(crop(vol, bbox), self.encoder, self.pool_size))
        return distance_matrix(embeddings, weights), bbox

    def select(
        self,
        phases: Sequence[PathOrArray],
        mask: PathOrArray,
        side: Union[str, int] = "roi",
        phase_names: Optional[Sequence[str]] = None,
    ) -> CURIAtorResult:
        """Select the most informative third phase.

        Args:
            phases: Ordered phases ``[Pre, Post1, Post2, ...]`` as NIfTI paths or
                RAS ``[Z, Y, X]`` arrays, all on the same grid as ``mask``. Do not
                include non-DCE sequences (e.g. T2).
            mask: Label mask (e.g. BreastDivider output) as path or array.
            side: ``"left"``, ``"right"``, ``"roi"`` (any label) or an int label.
            phase_names: Optional names for reporting; defaults to file names or
                ``Pre, Post1, ...``.
        """
        if len(phases) < 2:
            raise ValueError("Need at least a pre-contrast and one post-contrast phase.")
        names = list(phase_names) if phase_names else _default_names(phases)

        dist, bbox = self.distance_matrix(phases, mask, side)
        fallback = len(phases) == 2
        if fallback:
            warnings.warn("Only Pre and Post1 given; reusing Post1 as the third phase.", stacklevel=2)
            selected = POST1
        else:
            selected = select_phase(dist)
        return CURIAtorResult(dist, selected, bbox, side, names, fallback)


def _default_names(phases: Sequence[PathOrArray]) -> list[str]:
    if all(isinstance(p, (str, PathLike)) for p in phases):
        return [str(p).replace("\\", "/").rsplit("/", 1)[-1] for p in phases]
    return ["Pre"] + [f"Post{i}" for i in range(1, len(phases))]
