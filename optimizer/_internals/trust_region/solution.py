# -*- coding: utf-8 -*-


import math
from typing import NamedTuple, Tuple

from optimizer._internals.common import typing
from optimizer._internals.structures.frozenstate import FrozenState
from optimizer._internals.structures.gradient import (
    Gradient,
    GradientCheck,
    make_gradient,
)
from overloads.typedefs import ndarray


class Solution(NamedTuple):
    fval: float
    x: ndarray
    grad: Gradient
    proj: typing.proj_t
    shifted_constr: typing.constraints_t
    hess_up_to_date: bool = False


def make_solution(
    iter: int,
    x: ndarray,
    g_infnorm: Tuple[float, float],
    state: FrozenState,
) -> Solution:
    fval = state.objective(x)
    # +inf被允许是因为可能存在极大的penalty
    assert not math.isnan(fval) and fval != -math.inf
    grad, proj = make_gradient(
        x,
        state.gradient,
        state.constraints,
        state.opts,
        check=GradientCheck(state.objective_np, iter, *g_infnorm),
    )
    A, b, lb, ub = state.constraints
    shifted_constr = (A, b - A @ x, lb - x, ub - x)
    return Solution(
        fval=fval,
        x=x,
        grad=grad,
        proj=proj,
        shifted_constr=shifted_constr,
    )
