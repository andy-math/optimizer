from typing import Callable, Optional, Tuple

import numpy
from optimizer._internals.findiff import findiff
from optimizer._internals.options import Trust_Region_Options
from overloads import difference
from overloads.typing import ndarray


class Grad_Check_Failed(BaseException):
    iter: int
    checker: Optional[Callable[[ndarray, ndarray], float]]
    analytic: ndarray
    findiff_: ndarray

    def __init__(
        self,
        iter: int,
        checker: Callable[[ndarray, ndarray], float],
        analytic: ndarray,
        findiff_: ndarray,
    ) -> None:
        self.iter = iter
        self.checker = checker
        self.analytic = analytic
        self.findiff_ = findiff_


def gradient_check(
    analytic: ndarray,
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    opts: Trust_Region_Options,
    f: Callable[[ndarray], ndarray],
    iter: int,
    gradient_infnorm: float,
    initial_gradient_infnorm: float,
) -> None:
    need_rel_check = (
        opts.check_iter is None or iter <= opts.check_iter
    ) and gradient_infnorm >= initial_gradient_infnorm * opts.check_rel

    if need_rel_check or opts.check_abs is not None:
        findiff_ = findiff(f, x, constraints)
        assert len(findiff_.shape) == 2 and findiff_.shape[0] == 1
        findiff_.shape = (findiff_.shape[1],)

    if need_rel_check:
        index = ~numpy.logical_and(
            numpy.abs(analytic) <= opts.tol_grad, numpy.abs(findiff_) <= opts.tol_grad
        )
        if difference.relative(analytic[index], findiff_[index]) > opts.check_rel:
            raise Grad_Check_Failed(iter, difference.relative, analytic, findiff_)
    if opts.check_abs is not None:
        if difference.absolute(analytic, findiff_) > opts.check_abs:
            raise Grad_Check_Failed(iter, difference.absolute, analytic, findiff_)
