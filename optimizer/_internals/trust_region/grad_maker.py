from typing import Callable, Tuple

from numerical.findiff import findiff
from numerical.typedefs import ndarray
from optimizer._internals.trust_region.grad_check import gradient_check
from optimizer._internals.trust_region.grad_cutoff import gradient_cutoff
from optimizer._internals.trust_region.options import Trust_Region_Options


def make_gradient(
    g: Callable[[ndarray], ndarray],
    f: Callable[[ndarray], ndarray],
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    iter: int,
    gradient_infnorm: float,
    initial_gradient_infnorm: float,
    opts: Trust_Region_Options,
    *,
    check: bool
) -> ndarray:
    analytic = g(x)
    if check:
        gradient_check(
            analytic,
            f,
            x,
            constraints,
            iter,
            gradient_infnorm,
            initial_gradient_infnorm,
            opts,
        )
    return gradient_cutoff(analytic, x, constraints, opts)


def make_hessian(
    g: Callable[[ndarray], ndarray],
    f: Callable[[ndarray], ndarray],
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    iter: int,
    gradient_infnorm: float,
    initial_gradient_infnorm: float,
    opts: Trust_Region_Options,
) -> ndarray:
    H = findiff(
        lambda x: make_gradient(
            g,
            f,
            x,
            constraints,
            iter,
            gradient_infnorm,
            initial_gradient_infnorm,
            opts,
            check=False,
        ),
        x,
        constraints,
    )
    H = (H.T + H) / 2.0
    return H
