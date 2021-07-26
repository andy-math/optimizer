import math
from typing import Tuple

import numpy
from overloads.typedefs import ndarray


def constr_preproc(
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray]
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    _eps = float(numpy.finfo(numpy.float64).eps)
    A, b, lb, ub = constraints
    # 行norm2归一化
    A_row_norm = numpy.maximum(numpy.sqrt(numpy.sum(A * A, axis=1)), math.sqrt(_eps))
    b = b / A_row_norm
    A_row_norm.shape = (A_row_norm.shape[0], 1)
    A = A / A_row_norm

    # 禁止出现全0的行
    assert not numpy.any(numpy.all(A == 0, axis=1))

    # 拼合A和b
    b.shape = (b.shape[0], 1)  # 这里inplace reshape是因为上文的除法已经产生了新的数组对象
    Ab: ndarray = numpy.concatenate((A, b), axis=1)  # type: ignore

    # 行去重
    Ab: ndarray = numpy.unique(Ab, axis=0)  # type: ignore

    return (Ab[:, :-1], Ab[:, -1], lb, ub)
