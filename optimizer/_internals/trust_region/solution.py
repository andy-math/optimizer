# -*- coding: utf-8 -*-


from typing import Final, Tuple

from optimizer._internals.common.findiff import findiff
from optimizer._internals.common.hessian import Hessian
from optimizer._internals.trust_region.frozenstate import FrozenState
from optimizer._internals.trust_region.grad_maker import (
    Gradient,
    GradientCheck,
    make_gradient,
)
from overloads.typing import ndarray


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

    def get_hessian(self) -> Hessian:
        self.hess_up_to_date = True
        H = findiff(
            lambda x: make_gradient(
                self.state.g, x, self.state.constraints, self.state.opts, check=None
            ).value,
            self.x,
            self.state.constraints,
        )
        return Hessian(
            H,
            max_times=self.x.shape[0]
            if self.state.opts.shaking == "x.shape[0]"
            else self.state.opts.shaking,
        )
