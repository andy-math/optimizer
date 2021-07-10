# -*- coding: utf-8 -*-


from typing import Final, Tuple

from optimizer._internals.common.findiff import findiff
from optimizer._internals.common.hessian import Hessian
from optimizer._internals.trust_region.active_set import ActiveSet
from optimizer._internals.trust_region.frozenstate import FrozenState
from optimizer._internals.trust_region.grad_maker import (
    Gradient,
    GradientCheck,
    get_raw_grad,
    make_gradient,
)
from overloads.typing import ndarray


class Solution:
    state: Final[FrozenState]
    fval: Final[float]
    x: Final[ndarray]
    activeSet: ActiveSet
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
        raw_grad = get_raw_grad(
            state.g,
            x,
            check=GradientCheck(
                state.f_np, iter, *g_infnorm, state.constraints, state.opts
            ),
        )
        self.activeSet = ActiveSet(raw_grad, x, state.constraints, state.opts)
        self.grad = make_gradient(raw_grad, self.activeSet)
        A, b, lb, ub = state.constraints
        self.shifted_constr = (A, b - A @ x, lb - x, ub - x)

    def get_hessian(self) -> Hessian:
        self.hess_up_to_date = True
        H = findiff(
            lambda x: make_gradient(
                get_raw_grad(self.state.g, x, check=None), self.activeSet
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
