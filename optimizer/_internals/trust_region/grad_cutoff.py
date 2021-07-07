from typing import Tuple

import numpy
from optimizer._internals.trust_region.options import Trust_Region_Options
from overloads.typing import ndarray


def gradient_cutoff(
    g: ndarray,
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    opts: Trust_Region_Options,
) -> ndarray:
    if opts.border_abstol is not None:
        lb, ub = constraints[-2:]
        g[numpy.logical_and(x - lb < opts.border_abstol, g > 0)] = 0.0
        g[numpy.logical_and(ub - x < opts.border_abstol, g < 0)] = 0.0
    return g
