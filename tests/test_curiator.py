import json

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from curiator import CURIAtor, select_phase, volume_distance
from curiator.cli import main
from curiator.distance import pool_depth, token_weights
from curiator.io import bounding_box, roi_from_mask


class MeanIntensityEncoder:
    """Deterministic stand-in for Curia: tokens are patch means of the raw slice."""

    grid_size = 4

    def __call__(self, slices):
        out = []
        for s in slices:
            t = torch.from_numpy(np.ascontiguousarray(s, dtype=np.float32))[None, None]
            t = torch.nn.functional.adaptive_avg_pool2d(t, self.grid_size)  # [1, 1, g, g]
            out.append(t.reshape(-1, 1).repeat(1, 3))  # [P, C=3]
        return torch.stack(out)


def make_case(shape=(12, 16, 16), n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[2:10, 2:14, 1:8] = 1  # left
    mask[2:10, 2:14, 8:15] = 2  # right
    base = rng.uniform(100, 200, size=shape).astype(np.float32)
    phases = [base]
    # Post3 (index 3) deviates most from Post1 inside the left breast.
    gains = [1.0, 1.2, 3.0, 1.4][:n_post]
    for g in gains:
        v = base.copy()
        v[mask == 1] *= g
        phases.append(v + 500.0)
    return phases, mask


def test_select_phase_excludes_pre_and_anchor():
    d = np.array([
        [0, 9, 9, 9],
        [9, 0, 2, 5],
        [9, 2, 0, 1],
        [9, 5, 1, 0],
    ], dtype=float)
    assert select_phase(d) == 3
    assert select_phase(d, candidates=[2]) == 2


def test_volume_distance_matches_naive():
    rng = np.random.default_rng(1)
    e_i, e_j = (torch.from_numpy(rng.normal(size=(3, 4, 5))).float() for _ in range(2))
    w = torch.from_numpy(rng.uniform(size=(3, 4))).float()
    a, b, wn = e_i.numpy(), e_j.numpy(), w.numpy()
    naive = np.mean([
        np.sqrt(sum(wn[d, p] * ((a[d, p] - b[d, p]) ** 2).mean() for p in range(4)) / wn[d].sum())
        for d in range(3)
    ])
    assert volume_distance(e_i, e_j, w) == pytest.approx(float(naive), rel=1e-5)
    assert volume_distance(e_i, e_i, w) == 0.0


def test_pool_depth_matches_avg_pool_on_tokens():
    x = torch.randn(10, 6, 3)
    ref = torch.nn.functional.avg_pool1d(x.permute(1, 2, 0), 4, 4, ceil_mode=True).permute(2, 0, 1)
    assert torch.allclose(pool_depth(x, 4), ref)
    assert pool_depth(x, 4).shape == (3, 6, 3)


def test_token_weights_range_and_shape():
    roi = np.ones((9, 20, 20), dtype=bool)
    roi[:, :, 10:] = False
    w = token_weights(roi, grid_size=4, pool_size=8)
    assert w.shape == (2, 16)
    assert float(w.min()) >= 0 and float(w.max()) <= 1


def test_roi_and_bbox():
    _, mask = make_case()
    assert bounding_box(roi_from_mask(mask, "left")) == (2, 10, 2, 14, 1, 8)
    assert bounding_box(roi_from_mask(mask, 3)) is None


def test_curiator_selects_most_distant_post():
    phases, mask = make_case()
    res = CURIAtor(encoder=MeanIntensityEncoder(), pool_size=4).select(phases, mask, side="left")
    assert res.selected_index == 3
    assert res.triplet_indices == (0, 1, 3)
    assert np.allclose(res.distance_matrix, res.distance_matrix.T)
    assert np.allclose(np.diag(res.distance_matrix), 0)
    json.dumps(res.to_dict())


def test_curiator_fallback_with_single_post():
    phases, mask = make_case(n_post=1)
    with pytest.warns(UserWarning):
        res = CURIAtor(encoder=MeanIntensityEncoder()).select(phases, mask, side="left")
    assert res.fallback and res.selected_index == 1


def test_shape_mismatch_raises():
    phases, mask = make_case()
    phases[2] = phases[2][:-1]
    with pytest.raises(ValueError, match="shape"):
        CURIAtor(encoder=MeanIntensityEncoder()).select(phases, mask, side="left")


def test_batch_cli(tmp_path, monkeypatch):
    phases, mask = make_case()
    img_dir, mask_dir, out = tmp_path / "imagesTr", tmp_path / "masks", tmp_path / "out"
    img_dir.mkdir(), mask_dir.mkdir()
    for ch, p in enumerate(phases):
        sitk.WriteImage(sitk.GetImageFromArray(p), str(img_dir / f"caseA_{ch:04d}.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(mask_dir / "caseA.nii.gz"))

    monkeypatch.setattr("curiator.core.CuriaEncoder", lambda **_: MeanIntensityEncoder())
    assert main(["batch", "-i", str(img_dir), "-m", str(mask_dir), "-o", str(out),
                 "--pool-size", "4", "--export"]) == 0

    left = json.loads((out / "distances" / "caseA_left.json").read_text())
    assert left["selected_phase"] == "caseA_0003.nii.gz"
    assert (out / "images" / "caseA_left_0002.nii.gz").exists()
    cropped = sitk.ReadImage(str(out / "images" / "caseA_left_0000.nii.gz"))
    assert cropped.GetSize() == (7, 12, 8)
    assert "caseA_right" in (out / "selections.csv").read_text()
