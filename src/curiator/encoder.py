"""Slice-wise feature extraction with the Curia radiology foundation model."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

import numpy as np
import torch

CURIA_MODEL_ID = "raidium/curia"


class SliceEncoder(Protocol):
    """Anything that maps a list of 2D slices to patch tokens.

    ``grid_size`` is the side length of the (square) patch-token grid, i.e.
    ``P == grid_size ** 2``.
    """

    grid_size: int

    def __call__(self, slices: Sequence[np.ndarray]) -> torch.Tensor:
        """Return patch tokens of shape ``[len(slices), P, C]`` on the CPU."""
        ...


def rescale_volume(vol: np.ndarray) -> np.ndarray:
    """Linearly rescale a volume so its 99.9th percentile maps to ~1000.

    Makes the pipeline independent of the storage scale (e.g. fastMRI breast is
    stored on a ~1e-3 scale, which otherwise makes every slice collapse to zero
    in the Curia processor, see :func:`normalize_slice`). A no-op in effect for
    data that is already on a typical MR intensity range.
    """
    hi = np.percentile(vol, 99.9)
    if hi <= 0 or not np.isfinite(hi):
        return vol
    return vol * (1000.0 / hi)


def normalize_slice(x: np.ndarray) -> np.ndarray:
    """Clip a slice to its [1, 99] percentile range and scale it to [0, 1].

    Note: the Curia image processor casts numpy inputs to ``int16``. Applied to
    the output of this function, that keeps only the pixels clipped at the 99th
    percentile, so Curia effectively embeds a map of the brightest (strongest
    enhancing) ~1% of each slice.
    """
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-6)


class CuriaEncoder:
    """Frozen Curia ViT-B/16 (DINOv2) returning per-slice patch tokens.

    Access to ``raidium/curia`` on the Hugging Face Hub may require accepting the
    model terms and authenticating (``hf auth login`` or the ``HF_TOKEN``
    environment variable).
    """

    def __init__(
        self,
        model_id: str = CURIA_MODEL_ID,
        device: Optional[str | torch.device] = None,
        batch_size: int = 32,
    ):
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.grid_size = self.model.config.image_size // self.model.config.patch_size

    @torch.no_grad()
    def __call__(self, slices: Sequence[np.ndarray]) -> torch.Tensor:
        out = []
        for i in range(0, len(slices), self.batch_size):
            batch = [normalize_slice(s) for s in slices[i : i + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            hidden = self.model(**inputs).last_hidden_state  # [B, 1 + P, C]
            out.append(hidden[:, 1:, :].float().cpu())  # drop the CLS token
        return torch.cat(out, dim=0)
