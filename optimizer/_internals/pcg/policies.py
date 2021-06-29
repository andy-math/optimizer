import math
from typing import Tuple

import numpy
from numpy import ndarray
from optimizer._internals.common.linneq import check, margin
from optimizer._internals.pcg.flag import Flag
from optimizer._internals.pcg.norm_l2 import norm_l2
from optimizer._internals.pcg.status import Status


def scale(g: ndarray, H: ndarray, _d: ndarray, delta: float) -> ndarray:
    """
    g = g0 + H @ base (H具有对称性)(此步在函数入口处patch)
    f = g @ x + 0.5 * (x @ H @ x)
    let x <- direct * alpha
    f = (g @ d) * alpha + 0.5 * (d @ H @ d) * ( alpha*alpha )

    当(d @ H @ d) > 0,最小值为(dHd)a + (gd) == 0的解
    也就是alpha <- -gd/dhd

    否则，小于0的hessian意味着往正反方向走，梯度都减小
    因此走梯度小于0的下降方向，走到信赖域边缘即可
    遇到大于0的梯度则走反方向
    """
    norm = norm_l2(_d)
    assert norm > 0
    d = _d / norm
    gd = float(g @ d)
    dHd = float(d @ H @ d)
    if dHd > 0:
        alpha = -gd / dHd
        return numpy.sign(alpha) * min(math.fabs(alpha), delta) * d  # type: ignore
    else:
        if gd <= 0:
            return delta * d  # type: ignore
        else:
            return (-delta) * d  # type: ignore


def subspace_decay(
    _g: ndarray,
    H: ndarray,
    _status: Status,
    _d: ndarray,
    _delta: float,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
) -> Status:

    _x = _status.x

    # scale patch
    g = _g if _x is None else _g + H @ _x

    # 勾股定理求出内接于大圆信赖域的、以base为圆心的小圆信赖域半径
    delta = _delta if _x is None else _delta - norm_l2(_x)

    # 如果小圆信赖域太小，或前进方向异常，直接返回，啥也不做
    if delta <= 0 or norm_l2(_d) <= 0:
        return _status

    # 使用精确的二次型方法确定最优缩放尺度
    d = scale(g, H, _d, delta)

    # 求出base处的约束切面上下限
    lb, ub = margin(_x if _x is not None else numpy.zeros(d.shape), constraints)

    # 初始化越界表
    eliminated = numpy.zeros(d.shape, dtype=numpy.bool_)

    for _ in range(d.shape[0]):
        # 更新越界表
        eliminated[numpy.logical_or(d < lb, ub < d)] = True
        # 对越界（过）的维度折半衰减
        d[eliminated] /= 2.0
        # 使用精确的二次型方法确定最优缩放尺度
        d = scale(g, H, d, delta)
        # 如果满足全部在界内，那么退出折半衰减
        if numpy.all(numpy.logical_and(lb <= d, d <= ub)):
            break
        # 如果全部都越界，也退出折半衰减
        if numpy.all(eliminated):
            break

    final_x = d if _x is None else _x + d

    # 如果折半衰减后不满足约束，放弃，返回None
    final_x.shape = (final_x.shape[0], 1)
    if not check(final_x, constraints):
        return Status(
            None, _status.iter, Flag.VIOLATE_CONSTRAINTS, _status.ill, _delta, _g, H
        )
    final_x.shape = (final_x.shape[0],)

    # 如果满足了约束，曾经衰减过，那么替换flag为“越界”
    if bool(numpy.any(eliminated)):
        return Status(
            final_x, _status.iter, Flag.VIOLATE_CONSTRAINTS, _status.ill, _delta, _g, H
        )

    # 否则返回预期的前进
    return Status(final_x, _status.iter, _status.flag, _status.ill, _delta, _g, H)
