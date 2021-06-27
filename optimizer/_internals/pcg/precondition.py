import numpy
from numerical.typedefs import ndarray


def hessian_precon(H: ndarray) -> ndarray:
    assert len(H.shape) == 2
    assert H.shape[0] == H.shape[1]
    # 取 max{ l2norm(col(H)), sqrt(eps) }
    # 预条件子 M = C.T @ C == diag(R)
    # 其中 H === H.T  =>  norm(col(H)) === norm(row(H))
    _eps = float(numpy.finfo(numpy.float64).eps)
    dnrms: ndarray = numpy.sqrt(numpy.sum(H * H, axis=1))
    R: ndarray = numpy.maximum(dnrms, numpy.sqrt(numpy.array([_eps])))
    return R
