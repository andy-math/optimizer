from __future__ import annotations

import math
from typing import Optional

from numpy import ndarray
from optimizer._internals.pcg.flag import Flag
from optimizer._internals.pcg.qpval import qpval
from overloads.shortcuts import assertNoInfNaN


class Status:
    x: Optional[ndarray]
    fval: Optional[float]
    iter: int
    flag: Flag
    size: Optional[float]
    ill: bool

    def __init__(
        self,
        x: Optional[ndarray],
        iter: int,
        flag: Flag,
        ill: bool,
        delta: float,
        g: ndarray,
        H: ndarray,
    ) -> None:
        fval: Optional[float]
        if x is not None:
            assertNoInfNaN(x)
            fval = qpval(g=g, H=H, x=x)
        else:
            fval = None
        self.x = x
        self.fval = fval
        self.iter = iter
        self.flag = flag
        self.size = None if x is None else math.sqrt(float(x @ x))
        self.ill = ill
        if self.size is not None:
            assert self.size / delta < 1.0 + 1e-6


def _compare(s1: Status, s2: Status) -> Status:
    if s1.x is None and s2.x is None:
        return s1
    elif s1.x is None:
        return s2
    elif s2.x is None:
        return s1
    else:
        assert s1.fval is not None and s1.size is not None
        assert s2.fval is not None and s2.size is not None
        if s1.fval < s2.fval or (s1.fval == s2.fval and s1.size <= s2.size):
            return s1
        else:
            return s2


def best_status(best: Status, *status: Optional[Status]) -> Status:
    for s in status:
        best = best if s is None else _compare(best, s)
    return best
