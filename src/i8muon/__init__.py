from i8muon._optimizer import Muon, muon_step, _adjust_lr, _GRAM_ASPECT_THRESHOLD
from i8muon._ns import NSInt8, recommend_coefficients, _DEFAULT_NS_COEFFS

__all__ = [
    "Muon",
    "muon_step",
    "NSInt8",
    "recommend_coefficients",
    "_DEFAULT_NS_COEFFS",
]
