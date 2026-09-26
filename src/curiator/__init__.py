"""CURIAtor: feature-space guided temporal phase selection for breast DCE-MRI."""

from .core import CURIAtor, CURIAtorResult, select_phase
from .distance import distance_matrix, volume_distance
from .encoder import CuriaEncoder

__version__ = "0.1.0"

__all__ = [
    "CURIAtor",
    "CURIAtorResult",
    "CuriaEncoder",
    "distance_matrix",
    "select_phase",
    "volume_distance",
]
