# -*- coding: utf-8 -*-


from typing import Tuple

import numpy

from overloads import bind_checker
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float
from overloads.typedefs import ndarray

_eps = float(numpy.finfo(numpy.float64).eps)


def _input_check(
    t: Tuple[
        ndarray, ndarray, ndarray, Tuple[ndarray, ndarray, ndarray, ndarray], float
    ]
) -> None:
    x, g, H, _, delta = t
    assert len(x.shape) == 2
    n, _ = x.shape
    assert g.shape == (n,)
    assert H.shape == (n, n)
    assertNoInfNaN(x)
    assertNoInfNaN(g)
    assertNoInfNaN(H)
    assertNoInfNaN_float(delta)


def _output_check_x(x: ndarray) -> None:
    assert len(x.shape) == 1
    assertNoInfNaN(x)


def _output_check_bool(_: bool) -> None:
    pass


def _output_check_int(_: int) -> None:
    pass


@bind_checker.bind_checker_5(
    input=_input_check,
    output=bind_checker.make_checker_3(
        _output_check_x, _output_check_bool, _output_check_int
    ),
)
def clip_solution(
    x: ndarray,
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Tuple[ndarray, bool, int]:
    """
    x: S个N-dim的列向量
    先 assert 它们都是单位向量
    - 根据a = argmin f(a): a * gx + a^2 * 0.5xHx获得缩放尺度
        xHx > 0: 凸问题，a = -gx/xHx
        xHx <= 0: 非凸问题, a = sign(-gx) * Inf
    - 使用delta卡缩放尺度
    - 使用 a * Ax <= b 卡缩放尺度:
        a <= min(b/(Ax))
        出现负数说明二者异号，a无界
        出现NaN, Inf说明Ax不影响界，a无界
        需要使用(1-1e-4)避免刚性撞击边界
    """
    # 单位向量断言
    norm_x = numpy.sqrt(numpy.sum(x * x, axis=0))
    assert norm_x.max() - 1 < numpy.sqrt(_eps)
    # 最优二次型缩放
    gx: ndarray = g @ x
    xHx = numpy.sum(x * (H @ x), axis=0)
    a = -gx / xHx
    a[xHx <= 0] = numpy.sign(-gx[xHx <= 0]) * numpy.inf
    a[gx == 0] = 0  # gx必须出现在xHx之后而不是之前，用于解决(gx is 0) * inf -> NaN的问题
    # delta
    a = numpy.sign(a) * numpy.minimum(numpy.abs(a), delta)
    # a * Ax <= b
    lhs: ndarray = numpy.concatenate(
        # -x <= -lb; x <= ub; Ax <= b
        (-x, x, constraints[0] @ x),
        axis=0,
    )
    rhs: ndarray = numpy.concatenate(
        # -x <= -lb; x <= ub; Ax <= b
        (-constraints[2], constraints[3], constraints[1])
    )
    bound: ndarray = numpy.abs(rhs.reshape(-1, 1) / lhs)
    bound[a * lhs <= 0] = numpy.inf

    bound = 0.5 * bound.min(axis=0)  # type: ignore
    violate = numpy.abs(a) > bound
    a = numpy.sign(a) * numpy.minimum(numpy.abs(a), bound)

    qpval = a * gx + 0.5 * ((a * a) * xHx)
    index = int(numpy.argmin(qpval))
    x = a[index] * x[:, index]
    # assert check(x, constraints)
    return x, bool(violate[index]), index
