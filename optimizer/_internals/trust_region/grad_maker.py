from typing import Callable, NamedTuple, Optional, Tuple

import numpy
from numpy import ndarray
from numpy.linalg import eig, pinv  # type: ignore
from optimizer._internals.common.findiff import findiff
from optimizer._internals.trust_region.grad_check import gradient_check
from optimizer._internals.trust_region.grad_cutoff import gradient_cutoff
from optimizer._internals.trust_region.options import Trust_Region_Options
from scipy.linalg import ldl  # type: ignore


class Gradient(NamedTuple):
    value: ndarray
    infnorm: float


class Hessian:
    # https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/
    value: ndarray
    norm2: ndarray
    normF: ndarray
    pinv: ndarray
    norm2_pinv: ndarray
    normF_pinv: ndarray
    norm2_ldl_pinv: ndarray
    normF_ldl_pinv: ndarray
    ill: bool

    def __init__(self, value: ndarray) -> None:

        value = (value.T + value) / 2.0

        e: ndarray
        v: ndarray
        e, v = eig(value)
        assert e.dtype == numpy.float64
        min_e = float(numpy.min(e))

        self.value = value
        self.norm2 = value + numpy.diag(numpy.full(e.shape, -min(min_e, 0.0)))
        self.normF = v @ numpy.diag(numpy.maximum(e, 0.0)) @ v.T
        self.ill = min_e < 0

        self.pinv = pinv(value)
        self.norm2_pinv = pinv(self.norm2)
        self.normF_pinv = pinv(self.normF)

        l, d, perm = ldl(self.norm2)
        self.norm2_ldl_pinv = pinv(numpy.sqrt(d) @ l[perm, :].T)
        l, d, perm = ldl(self.normF)
        self.normF_ldl_pinv = pinv(numpy.sqrt(d) @ l[perm, :].T)


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
