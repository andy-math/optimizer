from typing import Callable, NamedTuple, Tuple

from optimizer._internals.common import typing
from optimizer._internals.common.norm import norm_inf
from optimizer._internals.trust_region.active_set import active_set
from optimizer._internals.trust_region.grad_check import gradient_check
from optimizer._internals.trust_region.options import Trust_Region_Options
from overloads.typedefs import ndarray


class Gradient(NamedTuple):
    value: ndarray
    infnorm: float


class GradientCheck(NamedTuple):
    f: Callable[[ndarray], ndarray]
    iter: int
    gradient_infnorm: float
    initial_gradient_infnorm: float


def make_gradient(
    x: ndarray,
    g: Callable[[ndarray], ndarray],
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    opts: Trust_Region_Options,
    *,
    check: GradientCheck,
) -> Tuple[Gradient, typing.proj_t]:
    analytic = g(x)
    gradient_check(analytic, x, constraints, opts, *check)
    proj = active_set(analytic, x, constraints, opts.border_abstol)
    return Gradient(analytic, norm_inf(proj @ analytic)), proj
