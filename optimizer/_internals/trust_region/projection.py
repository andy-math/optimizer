import math
from typing import List, Tuple

import numpy
from optimizer._internals.common import typing
from optimizer._internals.common.linneq import margin
from overloads.typedefs import ndarray

"""
lemma 1: 当v是正交的一组单位化列向量，
从g中取出它们的分量的方式是：
inner_product = g @ v
self_product = sum(v * v, axis=0)
alpha = inner_product / self_product
result = v @ alpha
其中，由单位化，self_product === 1
因此 alpha === inner_product
result = v @ v.T @ g
"""


def projection(
    g: ndarray,
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    border_abstol: float,
) -> typing.proj_t:
    fixing: List[ndarray] = []
    """
    第一段：向fixing中添加A中的行
    """
    _lb = numpy.full(x.shape, -numpy.inf)
    _ub = numpy.full(x.shape, numpy.inf)
    A, b, _, _ = constraints
    # A@g < 0，当沿着相反于g的方向，也就是 -g 方向走时，A@(-g)上升，到达上界b
    neq_violate = A @ (-g) > 0
    A, b = A[neq_violate, :], b[neq_violate]
    grad_pos: numpy.ndarray[Tuple[int, ...], numpy.dtype[numpy.bool_]] = g > 0
    grad_neg: numpy.ndarray[Tuple[int, ...], numpy.dtype[numpy.bool_]] = g < 0
    grad_nonzero = numpy.logical_or(grad_pos, grad_neg)
    border = numpy.zeros(g.shape)
    for i in range(A.shape[0]):
        _A, _b = A[[i], :], b[[i]]
        lb, ub = margin(x, (_A, _b, _lb, _ub))
        border[grad_pos] = -lb[grad_pos]  # 正的梯度导致数值减小
        border[grad_neg] = ub[grad_neg]  # 负的梯度导致数值变大
        if border[grad_nonzero].min() <= border_abstol:
            fixing.append(_A)
    """
    第二段：向fixing中添加lb或ub的实现
    """
    _, _, lb, ub = constraints
    assert g.shape == x.shape == lb.shape == ub.shape
    selector = numpy.zeros((1, x.shape[0]))
    for i in range(x.shape[0]):
        if g[i] > 0 and x[i] - lb[i] <= border_abstol:  # 正梯度导致数值减小
            selector[0, i] = -1.0  # 限制lb，系数为负
            fixing.append(selector.copy())  # 引用穿透
        elif g[i] < 0 and ub[i] - x[i] <= border_abstol:  # 负梯度导致数值变大
            selector[0, i] = 1.0  # 限制ub，系数为正
            fixing.append(selector.copy())  # 引用穿透
        selector[0, i] = 0.0  # 清除归零，此处不可短路continue
    """
    第三段：从特征值分解里提取0特征向量
    """
    e: ndarray
    v: ndarray
    if not len(fixing):
        return numpy.eye(g.shape[0])  # type: ignore
    _eps = float(numpy.finfo(numpy.float64).eps)
    fixA: ndarray = numpy.concatenate(fixing, axis=0)  # type: ignore
    e, v = numpy.linalg.eigh(fixA.T @ fixA)  # type: ignore

    zero_eigenvalues = numpy.abs(e) <= math.sqrt(_eps)
    zeros = numpy.sum(zero_eigenvalues)
    if not zeros:  # 未找到有效的基，只能返回全0
        return typing.proj_t(numpy.zeros((g.shape[0], g.shape[0])))
    v = v[:, zero_eigenvalues]
    return typing.proj_t(v @ v.T)
