# -*- coding: utf-8 -*-


import math
from typing import Tuple

import numpy
import scipy.optimize  # type: ignore

from optimizer._internals.common.linneq import constraint_check
from optimizer._internals.common.norm import norm_l2, safe_normalize
from optimizer._internals.quad_prog import flag, status
from optimizer._internals.quad_prog.circular_interp import circular_interp
from optimizer._internals.quad_prog.clip_solution import clip_solution
from optimizer._internals.quad_prog.quad_eval import QuadEvaluator
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


def no_check_QPeval(_: QuadEvaluator) -> None:
    pass


def no_check_Flag(_: Flag) -> None:
    pass


@bind_checker.bind_checker_2(
    input=bind_checker.make_checker_2(no_check_QPeval, assertNoInfNaN_float),
    output=bind_checker.make_checker_2(assertNoInfNaN, no_check_Flag),
)
def _implimentation(qpval: QuadEvaluator, delta: float) -> Tuple[ndarray, Flag]:
    g, H = qpval.g, qpval.H
    if norm_l2(g) < math.sqrt(_eps):
        return -g, Flag.INTERIOR

    e: ndarray
    v: ndarray
    e, v = numpy.linalg.eigh(H)  # type: ignore
    min_lambda = float(e.min())
    vg: ndarray = -g @ v

    s: ndarray
    if min_lambda > 0:
        s = v @ (vg / e)
        if norm_l2(s) <= delta:
            return s, Flag.INTERIOR

    flag: Flag = Flag.BOUNDARY

    def secular(lambda_: float) -> float:
        if min_lambda + lambda_ <= 0:
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
    e = e + lambda_
    assert not numpy.any(e < 0)
    if numpy.any(e == 0):
        flag = Flag.FATAL
        e[e == 0] = _eps
    s = v @ (vg / e)
    return delta * safe_normalize(s), flag


def _pcg_output_check(output: Status) -> None:
    pass


@dyn_signature
@bind_checker.bind_checker_3(
    input=bind_checker.make_checker_3(
        no_check_QPeval, constraint_check, assertNoInfNaN_float
    ),
    output=_pcg_output_check,
)
def quad_prog(
    qpval: QuadEvaluator,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Status:
    g, H = qpval.g, qpval.H
    d, flag = _implimentation(qpval, delta)
    x_clip, violate = clip_solution(circular_interp(-g, d), g, H, constraints, delta)
    if violate:
        flag = Flag.CONSTRAINT
    x_g, _ = clip_solution(
        safe_normalize(-g).reshape((-1, 1)), g, H, constraints, delta
    )
    x_d, _ = clip_solution(safe_normalize(d).reshape((-1, 1)), g, H, constraints, delta)
    assert qpval(x_clip) <= qpval(x_g) + 1e-6
    assert qpval(x_clip) <= qpval(x_d) + 1e-6
    return Status(x_clip, 0, flag, delta, qpval)
