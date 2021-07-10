import math
from typing import List, Tuple

import numpy
from optimizer._internals.common.linneq import margin
from overloads.typing import ndarray

"""
lemma 1: 当v是正交的一组单位化列向量，
从g中取出它们的分量的方式是：
inner_product = g @ v
self_product = sum(v * v, axis=0)
alpha = inner_product / self_product
result = v @ alpha
其中，由单位化，self_product === 1
因此 alpha === inner_product
result = v @ (g @ v)
"""


def active_set(
    g: ndarray,
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    border_abstol: float,
) -> ndarray:
    fixing: List[ndarray] = []
    """
    第一段：向fixing中添加A中的行
    """
    _lb = numpy.full(x.shape, -numpy.inf)
    _ub = numpy.full(x.shape, numpy.inf)
    A, b, _, _ = constraints
    # A@g < 0，当沿着相反于g的方向，也就是 -g 方向走时，A@(-g)上升，到达上界b
    Ag = A @ g < 0
    A, b = A[Ag, :], b[Ag]
    border = numpy.zeros(g.shape)
    for i in range(A.shape[0]):
        lb, ub = margin(x, (A[(i,), :], b[(i,)], _lb, _ub))
        border[g > 0] = -lb[g > 0]  # 正的梯度导致数值减小
        border[g < 0] = ub[g < 0]  # 负的梯度导致数值变大
        if border[g != 0].min() <= border_abstol:
            fixing.append(A[(i,), :])
    """
    第二段：向fixing中添加lb或ub的实现
    """
    _, _, lb, ub = constraints
    assert g.shape == x.shape == lb.shape == ub.shape
    selector = numpy.zeros(x.shape)  # 只开一个selector，用reshape副本防止引用穿透
    for i in range(x.shape[0]):
        if g[i] > 0 and x[i] - lb[i] <= border_abstol:  # 正梯度导致数值减小
            selector[i] = -1.0  # 限制lb，系数为负
            fixing.append(selector.reshape(1, -1))
        elif g[i] < 0 and ub[i] - x[i] <= border_abstol:  # 负梯度导致数值变大
            selector[i] = 1.0  # 限制ub，系数为正
            fixing.append(selector.reshape(1, -1))
        selector[i] = 0.0  # 清除归零，此处不可短路continue
    """
    第三段：从特征值分解里提取0特征向量
    """
    e: ndarray
    v: ndarray
    if not len(fixing):
        return g.copy()  # 防止引用穿透
    _eps = float(numpy.finfo(numpy.float64).eps)
    fixA: ndarray = numpy.concatenate(fixing, axis=0)  # type: ignore
    e, v = numpy.linalg.eig(fixA.T @ fixA)  # type: ignore
    if e.dtype.type != numpy.float64:
        e = e.real
    # 比较0特征值是否过半，决定逼近策略
    zero_eigenvalues = numpy.abs(e) <= math.sqrt(_eps)
    zeros = numpy.sum(zero_eigenvalues)
    if not zeros:  # 未找到有效的基，只能返回全0
        return numpy.zeros(g.shape)
    primal = zeros <= zero_eigenvalues.shape[0] // 2
    if primal:  # 零特征值较少，正向approach
        v = v[:, zero_eigenvalues]
    else:
        v = v[:, ~zero_eigenvalues]
    approach: ndarray = v @ (g @ v)  # lemma 1
    if primal:
        return approach  # 直接返回主0基
    else:
        return g - approach  # 去掉对偶的非0基，得到0基
