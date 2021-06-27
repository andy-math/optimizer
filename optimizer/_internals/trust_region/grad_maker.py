from typing import Callable, NamedTuple, Optional, Tuple

import numpy
from numerical.findiff import findiff
from numerical.typedefs import ndarray
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


def make_hessian(
    g: Callable[[ndarray], ndarray],
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    opts: Trust_Region_Options,
) -> ndarray:
    H = findiff(
        lambda x: make_gradient(g, x, constraints, opts, check=None).value,
        x,
        constraints,
    )
    H = (H.T + H) / 2.0
    return H