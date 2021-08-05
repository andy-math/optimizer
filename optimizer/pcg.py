# -*- coding: utf-8 -*-


import math
from typing import Optional, Tuple

import numpy
import scipy.optimize  # type: ignore

from optimizer._internals.common.linneq import constraint_check
from optimizer._internals.common.norm import norm_l2, safe_normalize
from optimizer._internals.pcg import flag, status
from optimizer._internals.pcg.circular_interp import circular_interp
from optimizer._internals.pcg.clip_solution import clip_solution
from optimizer._internals.pcg.qpval import QuadEvaluator
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float
from overloads.typedefs import ndarray

Flag = flag.Flag
Status = status.Status
_eps = float(numpy.finfo(numpy.float64).eps)


N = dyn_typing.SizeVar()
nConstraints = dyn_typing.SizeVar()

"""
动态类型签名
"""
dyn_signature = dyn_typing.dyn_check_3(
    input=(
        dyn_typing.Class(QuadEvaluator),
        dyn_typing.Tuple(
            (
                dyn_typing.NDArray(numpy.float64, (nConstraints, N)),
                dyn_typing.NDArray(numpy.float64, (nConstraints,)),
                dyn_typing.NDArray(numpy.float64, (N,)),
                dyn_typing.NDArray(numpy.float64, (N,)),
            )
        ),
        dyn_typing.Float(),
    ),
    output=dyn_typing.Class(Status),
)


def no_check(_: QuadEvaluator) -> None:
    pass


@bind_checker.bind_checker_3(
    input=bind_checker.make_checker_3(no_check, constraint_check, assertNoInfNaN_float),
    output=assertNoInfNaN,
)
def _implimentation(
    qpval: QuadEvaluator,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> ndarray:
    g, H = qpval.g, qpval.H
    if norm_l2(g) < math.sqrt(_eps):
        return -g

    e: ndarray
    v: ndarray
    e, v = numpy.linalg.eig(H)  # type: ignore
    e = e.real
    min_lambda = float(e.min())
    vg: ndarray = -g @ v

    s: ndarray
    if min_lambda > 0:
        s = v @ (vg / e)
        if norm_l2(s) <= delta:
            return s

    def secular(lambda_: float) -> float:
        if min_lambda + lambda_ == 0:
            return 1 / delta
        alpha: ndarray = vg / (e + lambda_)
        return (1 / delta) - (1 / norm_l2(alpha))

    def init_guess() -> Tuple[float, float]:
        a = -min_lambda if min_lambda < 0 else 0
        assert secular(a) >= 0
        dx = a / 2
        if not a:
            dx = 1 / 2
        while secular(a + dx) > 0:
            dx *= 2
        return (a, a + dx)

    lambda_ = scipy.optimize.brentq(
        secular, *init_guess(), maxiter=2 ** 31 - 1, disp=False
    )
    s = v @ (vg / (e + lambda_))
    return delta * safe_normalize(s)


def clip_direction(
    x: ndarray,
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
    *,
    basement: Optional[ndarray] = None
) -> ndarray:
    A, b, lb, ub = constraints
    if basement is not None:
        delta = numpy.sqrt(delta * delta - basement @ basement)
        assert numpy.isfinite(delta)
        g = g + H @ basement
        b = b - A @ basement
        lb = lb - basement
        ub = ub - basement
    x = safe_normalize(x).reshape((-1, 1))
    return clip_solution(x, g, H, (A, b, lb, ub), delta)


def _pcg_output_check(output: Status) -> None:
    pass


@dyn_signature
@bind_checker.bind_checker_3(
    input=bind_checker.make_checker_3(no_check, constraint_check, assertNoInfNaN_float),
    output=_pcg_output_check,
)
def pcg(
    qpval: QuadEvaluator,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Status:
    g, H = qpval.g, qpval.H
    d = _implimentation(qpval, constraints, delta)
    x = circular_interp(-g, d)
    x_clip = clip_solution(x, g, H, constraints, delta)
    x_g = clip_direction(-g, g, H, constraints, delta)
    x_d = clip_direction(d, g, H, constraints, delta)
    x_lstsq = clip_direction(
        numpy.linalg.lstsq(H, -g, rcond=None)[0],  # type: ignore
        g,
        H,
        constraints,
        delta,
    )
    assert qpval(x_clip) <= qpval(x_g) + 1e-6
    assert qpval(x_clip) <= qpval(x_d) + 1e-6
    if qpval(x_clip) <= qpval(x_lstsq):
        return Status(x_clip, 0, flag.Flag.POLICY_ONLY, delta, qpval)
    else:
        return Status(x_lstsq, 0, flag.Flag.POLICY_ONLY, delta, qpval)
