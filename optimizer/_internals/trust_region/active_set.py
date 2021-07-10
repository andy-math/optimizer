import math
from typing import List, Tuple

import numpy
from optimizer._internals.common.linneq import margin
from overloads.typing import ndarray

"""
对每一个当前的sol计算有效集A
对A.T @ A分解得到0特征值列向量v, v是正交矩阵（的一些列）
对从此处产生的grad, cut-off策略使用如下：
g = v @ [(g@v)/sum(v*v,axis=0)].T
其中，norm2(v)^2 === 1 (单位化)，上式简化为
g = v @ (g@v).T
进一步可以简化为
g = v @ v.T @ g
因此，Type`ActiveSet`定义为[v @ v.T]
当特征值全为0时，由正交矩阵，上式变为：
g = eye @ g
当特征值全不为0时，定义[v @ v.T]为0，简记为None
"""


def active_set(
    g: ndarray,
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    border_abstol: float,
) -> ndarray:
    fixing: List[ndarray] = []
    lb = numpy.full(x.shape, -numpy.inf)
    ub = numpy.full(x.shape, numpy.inf)
    A, b, _, _ = constraints
    assert A.shape[0] == b.shape[0]
    for i in range(A.shape[0]):
        A_row = A[[i], :]
        if A_row @ g >= 0:
            continue  # A@g >= 0，因此沿-g方向走，A@(-g)下降，达不到上界b
        lh, uh = margin(x, (A_row, b[[i]], lb, ub))
        border = numpy.zeros(g.shape)
        border[g > 0] = -lh[g > 0]  # 正的梯度导致数值减小
        border[g < 0] = uh[g < 0]  # 负的梯度导致数值变大
        if numpy.any(border[g != 0] <= border_abstol):
            fixing.append(A[i, :])  # 此处不用预制row是为了保持1D一致性，见下
    _, _, lb, ub = constraints
    assert g.shape == x.shape == lb.shape == ub.shape
    arange = numpy.arange(x.shape[0])
    for i in range(x.shape[0]):
        if g[i] > 0:  # 正梯度导致数值减小
            if x[i] - lb[i] <= border_abstol:
                fixing.append(-(arange == i).astype(numpy.float64))  # 限制lb
        elif g[i] < 0:  # 负梯度导致数值变大
            if ub[i] - x[i] <= border_abstol:
                fixing.append((arange == i).astype(numpy.float64))  # 限制ub
    if len(fixing):
        _eps = float(numpy.finfo(numpy.float64).eps)
        fixA = numpy.stack(fixing, axis=0)
        assert fixA.shape[1] == x.shape[0]
        e: ndarray
        v: ndarray
        e, v = numpy.linalg.eig(fixA.T @ fixA)  # type: ignore
        if e.dtype.type != numpy.float64:
            e = e.real
        v = v[:, numpy.abs(e) <= math.sqrt(_eps)]
        if not v.shape[1]:
            return numpy.zeros(g.shape)
        else:
            return v @ v.T @ g
    else:
        return g.copy()
