import math

import numpy
from optimizer._internals.pcg.norm_l2 import norm_l2
from overloads.typing import ndarray


def hessian_precon(H: ndarray) -> ndarray:
    assert len(H.shape) == 2
    assert H.shape[0] == H.shape[1]
    # 取 max{ l2norm(col(H)), sqrt(eps) }
    # 预条件子 M = C.T @ C == diag(R)
    # 其中 H === H.T  =>  norm(col(H)) === norm(row(H))
    _err = math.sqrt(float(numpy.finfo(numpy.float64).eps))
    dnrms: ndarray = numpy.sqrt(numpy.sum(H * H, axis=1))
    if numpy.any(numpy.isinf(dnrms)):  # 若l2计算过程中不可避免产生inf，那么使用inf范数代替之
        dnrms = numpy.abs(H).max(axis=1)
    R: ndarray = numpy.maximum(dnrms, _err)
    return R


def gradient_precon(g: ndarray) -> ndarray:
    assert len(g.shape) == 1
    R: ndarray = numpy.full(g.shape, max(norm_l2(g), 1.0))
    return R
