# -*- coding: utf-8 -*-


import math
from typing import Final, Tuple, cast

from optimizer._internals.common.findiff import findiff
from optimizer._internals.common.gradient import Gradient
from optimizer._internals.common.hessian import Hessian
from optimizer._internals.trust_region.frozenstate import FrozenState
from optimizer._internals.trust_region.grad_maker import GradientCheck, make_gradient
from overloads.typedefs import ndarray


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
        fval = state.f(x)
        # +inf被允许是因为可能存在极大的penalty
        assert not math.isnan(fval) and fval != -math.inf
        self.state = state
        self.fval = fval
        self.x = x
        self.grad = make_gradient(
            state.g,
            x,
            state.constraints,
            state.opts,
            check=GradientCheck(state.f_np, iter, *g_infnorm),
        )
        A, b, lb, ub = state.constraints
        self.shifted_constr = (A, b - A @ x, lb - x, ub - x)

    def get_hessian(self) -> Hessian:
        self.hess_up_to_date = True
        H = findiff(self.state.g, self.x, self.state.constraints)
        H = cast(ndarray, H.T + H) / 2.0
        VVT = self.grad.VVT
        H = VVT @ H @ VVT
        return Hessian(
            H,
            max_times=self.x.shape[0]
            if self.state.opts.shaking == "x.shape[0]"
            else self.state.opts.shaking,
        )
