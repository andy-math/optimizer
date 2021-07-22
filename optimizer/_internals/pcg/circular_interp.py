# -*- coding: utf-8 -*-


from typing import Tuple

import numpy
from optimizer._internals.common.norm import norm_l2, safe_normalize
from overloads import bind_checker
from overloads.shortcuts import assertNoInfNaN
from overloads.typing import ndarray

_eps = float(numpy.finfo(numpy.float64).eps)


def _input_check(t: Tuple[ndarray, ndarray]) -> None:
    direct1, direct2 = t
    assert len(direct1.shape) == 1
    assert len(direct2.shape) == 1
    assert direct1.shape == direct2.shape
    assertNoInfNaN(direct1)
    assertNoInfNaN(direct2)


def _output_check(x: ndarray) -> None:
    assert len(x.shape) == 1
    assertNoInfNaN(x)


@bind_checker.bind_checker_2(input=_input_check, output=_output_check)
def circular_interp(direct1: ndarray, direct2: ndarray) -> ndarray:
    num: int = 100

    direct1 = safe_normalize(direct1)
    direct2 = safe_normalize(direct2)

    d1_l2norm = norm_l2(direct1)
    d2_l2norm = norm_l2(direct2)

    # 两个都没有时，返回任意一个都是正确的零向量
    if not d2_l2norm:
        return direct1.reshape(-1, 1)
    if not d1_l2norm:
        return direct2.reshape(-1, 1)

    cos: float = float(direct1 @ direct2) / (d1_l2norm * d2_l2norm)
    if numpy.abs(cos - 1) < numpy.sqrt(_eps):
        return direct1.reshape(-1, 1)  # 相似度太高，返回任意一个都正确
    else:
        rad = numpy.linspace(0, numpy.arccos(cos), num=num)
        w1, w2 = numpy.cos(rad), numpy.sin(rad)
        direct2 = (direct2 - direct1 * w1[-1]) / w2[-1]  # 正交化
        x: ndarray = w1 * direct1.reshape(-1, 1) + w2 * direct2.reshape(-1, 1)
        return x
