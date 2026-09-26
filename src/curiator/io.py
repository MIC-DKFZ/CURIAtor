"""NIfTI loading, ROI masks and bounding-box cropping.

All arrays are returned in RAS orientation with numpy axis order ``[Z, Y, X]``
(SimpleITK convention), so slices along axis 0 are axial.
"""

from __future__ import annotations

from os import PathLike
from typing import Union

import numpy as np
import SimpleITK as sitk

PathOrArray = Union[str, PathLike, np.ndarray]

# Label convention of the BreastDivider model (https://github.com/MIC-DKFZ/BreastDivider).
BREASTDIVIDER_LABELS = {"left": 1, "right": 2}

BBox = tuple[int, int, int, int, int, int]  # (z0, z1, y0, y1, x0, x1), exclusive upper bounds


def load_nifti_ras(path: Union[str, PathLike]) -> tuple[sitk.Image, np.ndarray]:
    """Read an image and reorient it to RAS. Returns ``(sitk_image_ras, array[Z, Y, X])``."""
    img = sitk.ReadImage(str(path))
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAS")
    img_ras = orienter.Execute(img)
    return img_ras, sitk.GetArrayFromImage(img_ras)


def as_array(x: PathOrArray) -> np.ndarray:
    """Return ``x`` as a ``[Z, Y, X]`` array; paths are loaded and reoriented to RAS."""
    if isinstance(x, np.ndarray):
        return x
    return load_nifti_ras(x)[1]


def roi_from_mask(mask: np.ndarray, side: Union[str, int] = "roi") -> np.ndarray:
    """Binary ROI from a label mask.

    ``side`` is ``"left"`` / ``"right"`` (BreastDivider labels 1 / 2), ``"roi"``
    (any non-zero voxel) or an explicit integer label.
    """
    if isinstance(side, str):
        if side == "roi":
            return mask > 0
        if side not in BREASTDIVIDER_LABELS:
            raise ValueError(f"Unknown side {side!r}; use 'left', 'right', 'roi' or an int label.")
        side = BREASTDIVIDER_LABELS[side]
    return mask == side


def bounding_box(roi: np.ndarray) -> BBox | None:
    """Minimal bounding box of the ``True`` voxels, or ``None`` if ``roi`` is empty."""
    if not np.any(roi):
        return None
    zz, yy, xx = np.where(roi)
    return (
        int(zz.min()), int(zz.max()) + 1,
        int(yy.min()), int(yy.max()) + 1,
        int(xx.min()), int(xx.max()) + 1,
    )


def crop(arr: np.ndarray, bbox: BBox) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    return arr[z0:z1, y0:y1, x0:x1]


def crop_to_file(src: Union[str, PathLike], bbox: BBox, out: Union[str, PathLike]) -> None:
    """Crop ``src`` (in RAS) to ``bbox`` and write it, preserving spacing and direction.

    The output is stored in RAS orientation; its origin is updated so it stays
    physically aligned with the source.
    """
    img, _ = load_nifti_ras(src)
    z0, z1, y0, y1, x0, x1 = bbox
    # SimpleITK indexes [x, y, z]; the bbox is in numpy [z, y, x] order.
    sitk.WriteImage(img[x0:x1, y0:y1, z0:z1], str(out))
