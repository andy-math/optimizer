# -*- coding: utf-8 -*-


from typing import Final, Tuple, cast

from optimizer._internals.findiff import findiff
from optimizer._internals.linneq import margin
from optimizer._internals.frozenstate import FrozenState
from optimizer._internals.grad_maker import (
    Gradient,
    GradientCheck,
    make_gradient,
)
from overloads.typing import ndarray


class Solution:
    state: Final[FrozenState]
    fval: Final[float]
    x: Final[ndarray]
    gradient: Final[Gradient]
    lower_bound: Final[ndarray]
    upper_bound: Final[ndarray]

    def __init__(
        self,
        iter: int,
        fval: float,
        x: ndarray,
        g_infnorm: Tuple[(float, float)],
        state: FrozenState,
    ) -> None:
        self.state = state
        self.fval = fval
        assert fval == state.f(x)
        self.x = x
        self.lower_bound, self.upper_bound = margin(x, state.constraints)
        self.gradient = make_gradient(
            state.g,
            x,
            state.constraints,
            state.opts,
            check=GradientCheck(state.f_np, iter, *g_infnorm),
        )

    def get_hessian(self) -> ndarray:
        self.hess_up_to_date = True
        H = findiff(
            lambda x: make_gradient(
                self.state.g, x, self.state.constraints, self.state.opts, check=None
            ).value,
            self.x,
            self.state.constraints,
        )
        return cast(ndarray, H.T + H) / 2.0
