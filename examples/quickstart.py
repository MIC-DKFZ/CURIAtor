"""Minimal CURIAtor example: select the third DCE phase for both breasts of one case.

    python examples/quickstart.py PRE POST1 POST2 ... --mask BREASTDIVIDER_MASK
"""

import argparse

from curiator import CURIAtor
from curiator.plotting import plot_distance_matrix

parser = argparse.ArgumentParser()
parser.add_argument("phases", nargs="+", help="Pre, Post1, Post2, ... (NIfTI)")
parser.add_argument("--mask", required=True, help="BreastDivider mask (1 = left, 2 = right)")
args = parser.parse_args()

curiator = CURIAtor()  # loads raidium/curia on first use
for side in ("left", "right"):
    result = curiator.select(args.phases, args.mask, side=side)
    pre, post1, selected = (result.phase_names[i] for i in result.triplet_indices)
    print(f"{side:>5}: Pre={pre}  Post1={post1}  CURIAtor={selected}")
    print("       distance to Post1:", result.distances_to_post1.round(3))
    plot_distance_matrix(result.distance_matrix, selected=result.selected_index, out=f"curiator_{side}.png")
