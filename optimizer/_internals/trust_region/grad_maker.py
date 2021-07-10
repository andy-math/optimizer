from typing import Callable, NamedTuple, Optional, Tuple

import numpy
from optimizer._internals.common.gradient import Gradient, RawGradient
from optimizer._internals.trust_region.active_set import ActiveSet
from optimizer._internals.trust_region.grad_check import gradient_check
from optimizer._internals.trust_region.options import Trust_Region_Options
from overloads.typing import ndarray


class GradientCheck(NamedTuple):
    f: Callable[[ndarray], ndarray]
    iter: int
    gradient_infnorm: float
    initial_gradient_infnorm: float


def make_gradient(
    g: Callable[[ndarray], ndarray],
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    opts: Trust_Region_Options,
    *,
    check: Optional[GradientCheck],
) -> Gradient:
    analytic = g(x)
    if check is not None:
        gradient_check(analytic, x, constraints, opts, *check)
    gradient = ActiveSet(RawGradient(analytic), x, constraints, opts).cutoff(analytic)
    return Gradient(gradient, float(numpy.abs(gradient).max()))
