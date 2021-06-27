import math
from typing import Optional, Tuple

import numpy
from numerical.linneq import check, margin
from numerical.typedefs import ndarray
from optimizer._internals.pcg.flags import PCG_Flag


def subspace_decay(
    base: ndarray,
    direct: ndarray,
    delta: float,
    alpha: Optional[float],
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    exit_flag: PCG_Flag,
) -> Tuple[Optional[ndarray], PCG_Flag]:
    # 勾股定理求出内接于大圆信赖域的、以base为圆心的小圆信赖域半径
    delta = math.sqrt(delta * delta - float(base @ base))

    # 如果小圆信赖域太小，或者sqrt(negative) -> NaN
    # （在浮点误差的情况下会这样），直接返回，啥也不做
    if math.isnan(delta) or delta == 0:
        return base, exit_flag

    # 当存在二次型最优前进距离时，至多前进至最优点，而不是信赖域边界
    delta = delta if alpha is None else min(delta, alpha)
    if math.isnan(delta) or delta == 0:
        return base, exit_flag

    # 尝试对前进方向归一化，如果前进方向异常，直接返回base，啥也不做
    norm = math.sqrt(float(direct @ direct))
    if norm > 0:
        direct = direct / norm
    else:
        return base, exit_flag

    direct = direct * delta  # 将前进方向对齐到小圆边界
    lb, ub = margin(base, constraints)  # 求出base处的约束切面上下限
    eliminated = numpy.zeros(direct.shape, dtype=numpy.bool_)  # 初始化越界表
    while True:
        # 更新越界表
        eliminated[numpy.logical_or(direct < lb, ub < direct)] = True
        # 对越界（过）的维度折半衰减
        direct[eliminated] /= 2.0
        # 尝试归一化，如果折半衰减后还有可观测模长的话
        norm = math.sqrt(float(direct @ direct))
        if norm > 0:
            direct = direct / norm
        # 重新缩放到小圆信赖域
        direct = direct * delta
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
