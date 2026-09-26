"""Command-line interface: ``curiator select`` and ``curiator batch``."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from .core import CURIAtor, CURIAtorResult
from .io import crop_to_file, load_nifti_ras, roi_from_mask

SIDES = ("left", "right", "roi")


def _sides(side: str) -> list[str]:
    return ["left", "right"] if side == "both" else [side]


def _export(result: CURIAtorResult, phases: list[Path], out_dir: Path, case_id: str) -> list[str]:
    """Write the cropped (Pre, Post1, selected) triplet as ``{case_id}_000{0,1,2}.nii.gz``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for channel, idx in enumerate(result.triplet_indices):
        out = out_dir / f"{case_id}_{channel:04d}.nii.gz"
        if not out.exists():
            crop_to_file(phases[idx], result.bbox, out)
        written.append(str(out))
    return written


def _plot(result: CURIAtorResult, out: Path, title: str) -> None:
    from .plotting import plot_distance_matrix

    # Default Pre / Post1 / ... labels; file names are too long for tick labels.
    plot_distance_matrix(result.distance_matrix, None, result.selected_index, out, title)


def cmd_select(args: argparse.Namespace) -> int:
    phases = [Path(p) for p in args.phases]
    curiator = CURIAtor(pool_size=args.pool_size, device=args.device, batch_size=args.batch_size)
    results = {}
    for side in _sides(args.side):
        res = curiator.select(phases, args.mask, side=side)
        results[side] = res.to_dict()
        print(f"[{side}] selected {res.selected_name} (index {res.selected_index}), "
              f"D to Post1 = {res.distances_to_post1[res.selected_index]:.4f}")
        if args.plot:
            plot = Path(args.plot)
            if args.side == "both":
                plot = plot.with_name(f"{plot.stem}_{side}{plot.suffix}")
            _plot(res, plot, side)
        if args.export_dir:
            case_id = f"{args.case_id}_{side}" if args.side == "both" else args.case_id
            results[side]["exported"] = _export(res, phases, Path(args.export_dir), case_id)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


def _find_cases(images: Path, ignore: set[int]) -> dict[str, list[Path]]:
    pattern = re.compile(r"^(?P<case>.+)_(?P<ch>\d{4})\.nii(\.gz)?$")
    cases: dict[str, dict[int, Path]] = defaultdict(dict)
    for f in sorted(images.iterdir()):
        m = pattern.match(f.name)
        if m and int(m["ch"]) not in ignore:
            cases[m["case"]][int(m["ch"])] = f
    return {c: [chs[k] for k in sorted(chs)] for c, chs in sorted(cases.items())}


def _find_mask(masks: Path, case: str) -> Path | None:
    for name in (f"{case}.nii.gz", f"{case}_0000.nii.gz", f"{case}.nii"):
        if (masks / name).exists():
            return masks / name
    return None


def cmd_batch(args: argparse.Namespace) -> int:
    images, masks, out = Path(args.images), Path(args.masks), Path(args.output)
    dist_dir = out / "distances"
    dist_dir.mkdir(parents=True, exist_ok=True)
    cases = _find_cases(images, set(args.ignore_channels))
    if not cases:
        print(f"No '<case>_XXXX.nii.gz' files found in {images}", file=sys.stderr)
        return 1

    curiator = CURIAtor(pool_size=args.pool_size, device=args.device, batch_size=args.batch_size)
    rows, failures = [], []
    for i, (case, phases) in enumerate(cases.items(), 1):
        mask = _find_mask(masks, case)
        if mask is None:
            failures.append((case, "missing mask"))
            continue
        mask_np = load_nifti_ras(mask)[1] if args.side == "both" else None
        for side in _sides(args.side):
            case_id = f"{case}_{side}" if args.side == "both" else case
            cache = dist_dir / f"{case_id}.json"
            try:
                if mask_np is not None and not roi_from_mask(mask_np, side).any():
                    failures.append((case_id, f"side {side} absent in mask"))
                    continue
                if cache.exists():
                    rec = json.loads(cache.read_text())
                else:
                    res = curiator.select(phases, mask_np if mask_np is not None else mask, side=side)
                    rec = res.to_dict()
                    if args.export:
                        rec["exported"] = _export(res, phases, out / "images", case_id)
                    cache.write_text(json.dumps(rec, indent=2))
                    if args.plot:
                        (out / "plots").mkdir(exist_ok=True)
                        _plot(res, out / "plots" / f"{case_id}.png", case_id)
            except Exception as e:  # keep going; report at the end
                failures.append((case_id, str(e)))
                continue
            rows.append({"case": case_id, "n_phases": len(rec["phases"]),
                         "selected_index": rec["selected_index"],
                         "selected_phase": rec["selected_phase"], "fallback": rec["fallback"]})
            print(f"[{i}/{len(cases)}] {case_id}: {rec['selected_phase']}")

    with open(out / "selections.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case", "n_phases", "selected_index", "selected_phase", "fallback"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} selections written to {out / 'selections.csv'}")
    if failures:
        print(f"{len(failures)} skipped:", file=sys.stderr)
        for case_id, why in failures:
            print(f"  {case_id}: {why}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="curiator", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--side", choices=(*SIDES, "both"), default="both",
                        help="BreastDivider side (1=left, 2=right), 'roi' for any label, or 'both' (default).")
    common.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu (default: auto).")
    common.add_argument("--batch-size", type=int, default=32, help="Slices per Curia forward pass.")
    common.add_argument("--pool-size", type=int, default=8, help="Depth pooling kernel (default: 8).")

    s = sub.add_parser("select", parents=[common], help="Select the third phase for one acquisition.")
    s.add_argument("phases", nargs="+", help="Ordered NIfTI phases: PRE POST1 [POST2 ...].")
    s.add_argument("-m", "--mask", required=True, help="Breast mask (e.g. BreastDivider output).")
    s.add_argument("--json", help="Write results (incl. distance matrix) to this JSON file.")
    s.add_argument("--plot", help="Save the distance matrix plot (PNG/PDF).")
    s.add_argument("--export-dir", help="Write the cropped (Pre, Post1, selected) triplet here.")
    s.add_argument("--case-id", default="case", help="Filename prefix for --export-dir.")
    s.set_defaults(func=cmd_select)

    b = sub.add_parser("batch", parents=[common], help="Run on an nnU-Net style folder of cases.")
    b.add_argument("-i", "--images", required=True,
                   help="Folder with <case>_XXXX.nii.gz; channel 0000=Pre, 0001=Post1, then later posts.")
    b.add_argument("-m", "--masks", required=True, help="Folder with <case>.nii.gz (or <case>_0000.nii.gz) masks.")
    b.add_argument("-o", "--output", required=True, help="Output folder.")
    b.add_argument("--ignore-channels", type=int, nargs="*", default=[],
                   help="Channel indices that are not DCE phases (e.g. 3 for T2 in ODELIA).")
    b.add_argument("--export", action="store_true", help="Also write cropped triplets to <output>/images.")
    b.add_argument("--plot", action="store_true", help="Also save distance matrix plots to <output>/plots.")
    b.set_defaults(func=cmd_batch)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
