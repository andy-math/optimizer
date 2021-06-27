import math
from typing import Optional, Tuple

import numpy
from numerical.linneq import check, margin
from numerical.typedefs import ndarray
from optimizer._internals.pcg.flags import PCG_Flag


def scale(g: ndarray, H: ndarray, x: ndarray, delta: float) -> ndarray:
    """
    g = g0 + H @ base (H具有对称性)(此步在函数入口处patch)
    f = g @ x + 0.5 * (x @ H @ x)
    let x <- direct * alpha
    f = (g @ d) * alpha + 0.5 * (d @ H @ d) * ( alpha*alpha )

    当(d @ H @ d) > 0,最小值为(dHd)a + (gd) == 0的解
    也就是alpha <- -gd/dhd

    否则，alpha在无穷远处，但近处可能不降反升
    """
    norm = math.sqrt(float(x @ x))
    if norm > 0:
        x = x / norm
    dHd = float(x @ H @ x)
    if dHd > 0:
        alpha = -float(g @ x) / dHd
        return min(alpha, delta) * x  # type: ignore
    else:
        xmax, xmin = delta * x, (-delta) * x
        vmax = float(g @ xmax + 0.5 * (xmax @ H @ xmax))
        vmin = float(g @ xmin + 0.5 * (xmin @ H @ xmin))
        assert vmax <= 0 or vmin <= 0  # 非凸情形下总有一侧是下降的
        if vmax <= vmin:
            return xmax  # type: ignore
        else:
            return xmin  # type: ignore


def subspace_decay(
    g: ndarray,
    H: ndarray,
    base: ndarray,
    direct: ndarray,
    delta: float,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    exit_flag: PCG_Flag,
) -> Tuple[Optional[ndarray], PCG_Flag]:
    g = g + H @ base  # scale patch

    # 勾股定理求出内接于大圆信赖域的、以base为圆心的小圆信赖域半径
    delta = math.sqrt(delta * delta - float(base @ base))

    # 如果小圆信赖域太小，或者sqrt(negative) -> NaN
    # （在浮点误差的情况下会这样），直接返回，啥也不做
    if math.isnan(delta) or delta == 0:
        return base, exit_flag

    # 如果前进方向异常，直接返回base，啥也不做
    norm = math.sqrt(float(direct @ direct))
    if math.isnan(norm) or norm == 0:
        return base, exit_flag

    direct = scale(g, H, direct, delta)  # 使用精确的二次型方法确定最优缩放尺度
    lb, ub = margin(base, constraints)  # 求出base处的约束切面上下限
    eliminated = numpy.zeros(direct.shape, dtype=numpy.bool_)  # 初始化越界表
    while True:
        # 更新越界表
        eliminated[numpy.logical_or(direct < lb, ub < direct)] = True
        # 对越界（过）的维度折半衰减
        direct[eliminated] /= 2.0
        # 使用精确的二次型方法确定最优缩放尺度
        direct = scale(g, H, direct, delta)
        # 如果满足全部在界内，那么退出折半衰减
        if numpy.all(numpy.logical_and(lb <= direct, direct <= ub)):
            break
        # 如果全部都越界，也退出折半衰减
        if numpy.all(eliminated):
            break

    base = base + direct
    # 如果折半衰减后不满足约束，放弃，返回None
    base.shape = (base.shape[0], 1)
    if not check(base, constraints):
        return None, PCG_Flag.VIOLATE_CONSTRAINTS
    base.shape = (base.shape[0],)

    # 如果满足了约束，曾经衰减过，那么替换flag为“越界”
    if bool(numpy.any(eliminated)):
        return base, PCG_Flag.VIOLATE_CONSTRAINTS

    # 否则返回预期的前进
    return base, exit_flag
