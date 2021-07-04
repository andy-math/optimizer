from typing import Callable, NamedTuple, Optional, Tuple

import numpy
from numpy import ndarray
from optimizer._internals.common.findiff import findiff
from optimizer._internals.common.hessian import Hessian
from optimizer._internals.trust_region.grad_check import gradient_check
from optimizer._internals.trust_region.grad_cutoff import gradient_cutoff
from optimizer._internals.trust_region.options import Trust_Region_Options


class Gradient(NamedTuple):
    value: ndarray
    infnorm: float


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
    gradient = gradient_cutoff(analytic, x, constraints, opts)
    return Gradient(gradient, float(numpy.max(numpy.abs(gradient))))
