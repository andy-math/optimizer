from typing import Callable, NamedTuple, Optional, Tuple

import numpy
from optimizer._internals.active_set import active_set
from optimizer._internals.grad_check import gradient_check
from optimizer._internals.options import Trust_Region_Options
from overloads.typing import ndarray


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
    gradient = active_set(analytic, x, constraints, opts.border_abstol)
    return Gradient(gradient, float(numpy.abs(gradient).max()))
