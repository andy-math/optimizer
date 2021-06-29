import math
from typing import Callable, NamedTuple, Optional, Tuple

import numpy
from numpy import ndarray
from numpy.linalg import cholesky, eig, pinv  # type: ignore
from optimizer._internals.common.findiff import findiff
from optimizer._internals.trust_region.grad_check import gradient_check
from optimizer._internals.trust_region.grad_cutoff import gradient_cutoff
from optimizer._internals.trust_region.options import Trust_Region_Options


class Gradient(NamedTuple):
    value: ndarray
    infnorm: float


class Hessian:
    # https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/
    value: ndarray
    pinv: ndarray
    normF: ndarray
    normF_chol: ndarray
    ill: bool

    def __init__(self, value: ndarray) -> None:
        value = (value.T + value) / 2.0
        e: ndarray
        v: ndarray
        e, v = eig(value)
        assert e.dtype == numpy.float64
        self.value = value
        self.ill = float(numpy.min(e)) < 0.0
        self.pinv = pinv(value)

        _eps = float(numpy.finfo(numpy.float64).eps)
        e = numpy.diag(numpy.maximum(e, math.sqrt(_eps)))
        self.normF = v @ e @ v.T
        self.normF_chol = cholesky(self.normF)


class GradientCheck(NamedTuple):
    f: Callable[[ndarray], ndarray]
    iter: int
    gradient_infnorm: float
    initial_gradient_infnorm: float


def make_gradient(
    g: Callable[[ndarray], ndarray],
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    opts: Trust_Region_Options,
    *,
    check: Optional[GradientCheck],
) -> Gradient:
    analytic = g(x)
    if check is not None:
        gradient_check(analytic, x, constraints, opts, *check)
    gradient = gradient_cutoff(analytic, x, constraints, opts)
    return Gradient(gradient, float(numpy.max(numpy.abs(gradient))))


def make_hessian(
    g: Callable[[ndarray], ndarray],
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    opts: Trust_Region_Options,
) -> Hessian:
    H = findiff(
        lambda x: make_gradient(g, x, constraints, opts, check=None).value,
        x,
        constraints,
    )
    return Hessian(H)
