import math
from typing import Final, Tuple

import numpy

from optimizer._internals.common.findiff import findiff
from optimizer._internals.structures.frozenstate import FrozenState
from optimizer._internals.trust_region.solution import Solution
from overloads.typedefs import ndarray


class Hessian:
    value: Final[ndarray]
    ill: Final[bool]
    times: int = 0
    max_times: Final[int]

    def __init__(self, value: ndarray, *, max_times: int) -> None:
        _err = math.sqrt(float(numpy.finfo(numpy.float64).eps))

        value = (value.T + value) / 2.0
        e: ndarray = numpy.linalg.eigh(value)[0]
        assert e.dtype.type == numpy.float64
        min_e = float(e.min())

        self.value = value
        self.ill = min_e < _err
        self.max_times = max_times


def make_hessian(sol: Solution, state: FrozenState) -> Tuple[Solution, Hessian]:
    sol = Solution(
        fval=sol.fval,
        x=sol.x,
        grad=sol.grad,
        proj=sol.proj,
        shifted_constr=sol.shifted_constr,
        hess_up_to_date=True,
    )
    H = findiff(state.gradient, sol.x, state.constraints)
    if state.opts.shaking == "x.shape[0]":
        max_times = sol.x.shape[0]
    else:
        max_times = state.opts.shaking
    return sol, Hessian(H, max_times=max_times)
