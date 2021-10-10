from typing import Callable, NamedTuple, Optional, Tuple

from optimizer._internals.common.norm import norm_inf
from optimizer._internals.trust_region.active_set import active_set
from optimizer._internals.trust_region.grad_check import gradient_check
from optimizer._internals.trust_region.options import Trust_Region_Options
from overloads.typedefs import ndarray


class Gradient(NamedTuple):
    value: ndarray
    infnorm: float
    VVT: ndarray


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
    VVT = active_set(analytic, x, constraints, opts.border_abstol)
    gradient: ndarray = VVT @ analytic
    return Gradient(gradient, norm_inf(gradient), VVT)
