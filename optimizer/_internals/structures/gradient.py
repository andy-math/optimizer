from typing import Final, NamedTuple, Tuple

from optimizer._internals.common import typing
from optimizer._internals.common.findiff import findiff
from optimizer._internals.common.norm import norm_inf
from optimizer._internals.structures.frozenstate import FrozenState
from optimizer._internals.trust_region.options import Trust_Region_Options
from optimizer._internals.trust_region.projection import projection
from overloads import difference
from overloads.typedefs import ndarray


class GradientCheck(NamedTuple):
    iter: int
    gradient_infnorm: float
    initial_gradient_infnorm: float


class Grad_Check_Failed(BaseException):
    iter: Final[int]
    error: Final[float]
    analytic: Final[ndarray]
    findiff_: Final[ndarray]

    def __init__(
        self,
        iter: int,
        error: float,
        analytic: ndarray,
        findiff_: ndarray,
    ) -> None:
        self.iter = iter
        self.error = error
        self.analytic = analytic
        self.findiff_ = findiff_


def gradient_check(
    analytic: ndarray, x: ndarray, state: FrozenState, check: GradientCheck
) -> None:
    iter: Final[int] = check.iter
    gradient_infnorm: Final[float] = check.gradient_infnorm
    initial_gradient_infnorm: Final[float] = check.initial_gradient_infnorm

    opts: Final[Trust_Region_Options] = state.opts
    need_rel_check = (
        opts.check_iter is None or iter <= opts.check_iter
    ) and gradient_infnorm >= initial_gradient_infnorm * opts.check_rel

    if need_rel_check or opts.check_abs is not None:
        findiff_ = findiff(state.objective_np, x, state.constraints)
        assert len(findiff_.shape) == 2 and findiff_.shape[0] == 1
        findiff_.shape = (findiff_.shape[1],)

        if need_rel_check:
            relerr = difference.relative(analytic, findiff_)
            if relerr > opts.check_rel:
                raise Grad_Check_Failed(iter, relerr, analytic, findiff_)
        if opts.check_abs is not None:
            abserr = difference.absolute(analytic, findiff_)
            if abserr > opts.check_abs:
                raise Grad_Check_Failed(iter, abserr, analytic, findiff_)


class Gradient(NamedTuple):
    value: ndarray
    infnorm: float


def make_gradient(
    x: ndarray,
    state: FrozenState,
    check: GradientCheck,
) -> Tuple[Gradient, typing.proj_t]:
    analytic = state.gradient(x)
    gradient_check(analytic, x, state, check)
    proj = projection(analytic, x, state.constraints, state.opts.border_abstol)
    return Gradient(analytic, norm_inf(proj @ analytic)), proj
