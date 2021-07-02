# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Final, Tuple

from numpy import ndarray
from optimizer._internals.common.hessian import Hessian
from optimizer._internals.trust_region import options
from optimizer._internals.trust_region.frozenstate import FrozenState
from optimizer._internals.trust_region.grad_maker import (
    Gradient,
    GradientCheck,
    make_gradient,
    make_hessian,
)

Trust_Region_Format_T = options.Trust_Region_Format_T
default_format = options.default_format
Trust_Region_Options = options.Trust_Region_Options


class HessianProxy:
    value: Hessian
    times: int = 0
    max_times: int

    def __init__(self, value: Hessian, max_times: int) -> None:
        self.value = value
        self.max_times = max_times


class Solution:
    state: Final[FrozenState]
    fval: Final[float]
    x: Final[ndarray]
    grad: Final[Gradient]
    shifted_constr: Final[Tuple[ndarray, ndarray, ndarray, ndarray]]
    hess_up_to_date: bool = False

    def __init__(
        self,
        iter: int,
        x: ndarray,
        g_infnorm: Tuple[(float, float)],
        state: FrozenState,
    ) -> None:
        self.state = state
        self.fval = state.f(x)
        self.x = x
        grad = make_gradient(
            state.g,
            x,
            state.constraints,
            state.opts,
            check=GradientCheck(state.f_np, iter, *g_infnorm),
        )
        self.grad = grad
        A, b, lb, ub = state.constraints
        self.shifted_constr = (A, b - A @ x, lb - x, ub - x)

    def get_hessian(self) -> HessianProxy:
        self.hess_up_to_date = True
        return HessianProxy(
            make_hessian(self.state.g, self.x, self.state.constraints, self.state.opts),
            self.x.shape[0]
            if self.state.opts.shaking == "x.shape[0]"
            else self.state.opts.shaking,
        )