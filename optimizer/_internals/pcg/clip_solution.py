# -*- coding: utf-8 -*-


from typing import Tuple

import numpy
from overloads import bind_checker
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float
from overloads.typing import ndarray

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


def _output_check(x: ndarray) -> None:
    assert len(x.shape) == 1
    assertNoInfNaN(x)


@bind_checker.bind_checker_5(input=_input_check, output=_output_check)
def clip_solution(
    x: ndarray,
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> ndarray:
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
    lhs: ndarray = numpy.concatenate(  # type: ignore
        # -x <= -lb; x <= ub; Ax <= b
        (-x, x, constraints[0] @ x),
        axis=0,
    )
    rhs: ndarray = numpy.concatenate(  # type: ignore
        # -x <= -lb; x <= ub; Ax <= b
        (-constraints[2], constraints[3], constraints[1])
    )
    bound: ndarray = rhs.reshape(-1, 1) / lhs
    bound[lhs == 0] = numpy.inf
    bound[bound < 0] = numpy.inf

    a = numpy.minimum(a, (1.0 - 1.0e-4) * bound.min(axis=0))

    qpval = a * gx + 0.5 * ((a * a) * xHx)
    index = int(numpy.argmin(qpval))
    return a[index] * x[:, index]  # type: ignore
