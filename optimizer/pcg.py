# -*- coding: utf-8 -*-


from typing import Optional, Tuple, cast

import numpy
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float
from overloads.typing import ndarray

from optimizer._internals.common.hessian import Hessian
from optimizer._internals.common.linneq import constraint_check
from optimizer._internals.pcg import flag, status

Flag = flag.Flag
Status = status.Status
_eps = float(numpy.finfo(numpy.float64).eps)


def _impl_input_check(
    input: Tuple[
        ndarray,
        ndarray,
        Tuple[ndarray, ndarray, ndarray, ndarray],
        float,
    ]
) -> None:
    g, H, constraints, delta = input
    assertNoInfNaN(g)
    assertNoInfNaN(H)
    constraint_check(constraints)
    assertNoInfNaN_float(delta)


def _impl_output_check(output: Tuple[Status, Optional[ndarray]]) -> None:
    status, direct = output
    if status.flag == Flag.RESIDUAL_CONVERGENCE:
        assert direct is None
    else:
        assert direct is not None
        assertNoInfNaN(direct)


@bind_checker.bind_checker_4(input=_impl_input_check, output=_impl_output_check)
def _implimentation(
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Tuple[Status, Optional[ndarray]]:
    def exit_(
        x: ndarray, d: Optional[ndarray], iter: int, flag: Flag
    ) -> Tuple[Status, Optional[ndarray]]:
        if iter != 0 or flag == Flag.RESIDUAL_CONVERGENCE:
            return Status(x, iter, flag, delta, g, H), d
        else:
            return Status(None, iter, flag, delta, g, H), d

    # 取 max{ l2norm(col(H)), sqrt(eps) }
    # 预条件子 M = C.T @ C == diag(R)
    # 其中 H === H.T  =>  norm(col(H)) === norm(row(H))
    R: ndarray = numpy.sqrt(numpy.sum(H * H, axis=1))
    if numpy.any(numpy.isinf(R)):  # 若l2计算过程中不可避免产生inf，那么使用inf范数代替之
        R = numpy.abs(H).max(axis=1)
    R = numpy.maximum(R, numpy.sqrt(_eps))

    (n,) = g.shape
    x: ndarray = numpy.zeros((n,))  # 目标点
    r: ndarray = -g  # 残差
    z: ndarray = r / R  # 归一化后的残差
    d: ndarray = z  # 搜索方向

    inner1: float = float(r.T @ z)

    for iter in range(n):
        # 残差收敛性检查
        if numpy.abs(z).max() < numpy.sqrt(_eps):
            return exit_(x, None, iter, Flag.RESIDUAL_CONVERGENCE)

        # 负曲率检查
        ww: ndarray = H @ d
        denom: float = float(d.T @ ww)
        if denom <= 0:
            return exit_(x, d, iter, Flag.NEGATIVE_CURVATURE)

        # 试探坐标点
        alpha: float = inner1 / denom
        x_new: ndarray = x + alpha * d

        # 目标点超出信赖域
        if x_new @ x_new > delta * delta:
            return exit_(x, d, iter, Flag.OUT_OF_TRUST_REGION)

        # 违反约束
        if (
            numpy.any(x_new < constraints[2])
            or numpy.any(x_new > constraints[3])
            or numpy.any(constraints[0] @ x_new > constraints[1])
        ):
            return exit_(x, d, iter, Flag.VIOLATE_CONSTRAINTS)

        # 更新坐标点
        x = x_new

        # 更新残差和右端项
        r = cast(ndarray, r - alpha * ww)
        z = cast(ndarray, r / R)

        # 更新搜索方向
        inner2: float = inner1
        inner1 = float(r.T @ z)
        beta: float = inner1 / inner2
        d = z + beta * d

    return exit_(x, None, iter, Flag.RESIDUAL_CONVERGENCE)


def clip_sol(
    x: ndarray,
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> ndarray:
    """
    x: S个N-dim的列向量
    先 assert 它们都是单位向量
    - 根据a = argmin f(a): a * gx + a^2 * 0.5xHx获得缩放尺度
        xHx > 0: 凸问题，a = -gx/xHx
        xHx <= 0: 非凸问题, a = sign(-gx) * Inf
    - 使用delta卡缩放尺度
    - 使用 a * Ax <= b 卡缩放尺度:
        a <= min(b/(Ax))
        出现负数说明二者异号，a无界
        出现NaN, Inf说明Ax不影响界，a无界
        需要使用(1-1e-4)避免刚性撞击边界
    """
    # 单位向量断言
    norm_x = numpy.sqrt(numpy.sum(x * x, axis=0))
    assert norm_x.max() - 1 < numpy.sqrt(_eps)
    # 最优二次型缩放
    gx: ndarray = g @ x
    xHx = numpy.sum(x * (H @ x), axis=0)
    a = -gx / xHx
    a[xHx <= 0] = numpy.sign(-gx[xHx <= 0]) * numpy.inf
    a[gx == 0] = 0
    # delta
    a = numpy.sign(a) * numpy.minimum(numpy.abs(a), delta)
    # a * Ax <= b
    lhs = numpy.concatenate(  # type: ignore
        (
            -x,  # -x <= -lb
            x,  # x <= ub
            constraints[0] @ x,  # Ax <= b
        ),
        axis=0,
    )
    rhs: ndarray = numpy.concatenate(  # type: ignore
        (
            -constraints[2],  # -x <= -lb
            constraints[3],  # x <= ub
            constraints[1],  # Ax <= b
        )
    )
    bound = rhs.reshape(-1, 1) / lhs
    bound[lhs == 0] = numpy.inf
    bound[bound < 0] = numpy.inf
    bound = (1.0 - 1.0e-4) * bound.min(axis=0)

    a = numpy.minimum(a, bound)

    qpval = a * gx + 0.5 * (a * xHx)
    index = int(numpy.argmin(qpval))
    return a[index] * x[:, index]  # type: ignore


def _pcg_input_check(
    input: Tuple[ndarray, Hessian, Tuple[ndarray, ndarray, ndarray, ndarray], float]
) -> None:
    g, _, constraints, delta = input
    assertNoInfNaN(g)
    constraint_check(constraints)
    assertNoInfNaN_float(delta)


def _pcg_output_check(output: Status) -> None:
    if output.x is not None:
        assert output.fval is not None
        assertNoInfNaN(output.x)
        assertNoInfNaN_float(output.fval)
    else:
        assert output.fval is None


N = dyn_typing.SizeVar()
nConstraints = dyn_typing.SizeVar()


@dyn_typing.dyn_check_4(
    input=(
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.Class(Hessian),
        dyn_typing.Tuple(
            (
                dyn_typing.NDArray(numpy.float64, (nConstraints, N)),
                dyn_typing.NDArray(numpy.float64, (nConstraints,)),
                dyn_typing.NDArray(numpy.float64, (N,)),
                dyn_typing.NDArray(numpy.float64, (N,)),
            )
        ),
        dyn_typing.Float(),
    ),
    output=dyn_typing.Class(Status),
)
@bind_checker.bind_checker_4(input=_pcg_input_check, output=_pcg_output_check)
def pcg(
    g: ndarray,
    H: Hessian,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Status:
    ret1, direct = _implimentation(g, H.value, constraints, delta)
    d = numpy.zeros(g.shape) if ret1.x is None else ret1.x
    if direct is not None:
        sqsize = delta * delta - d @ d
        if sqsize > 0:
            size = numpy.sqrt(sqsize)
            d = d + size * direct

    orig_g = g

    d_infnorm = numpy.abs(d).max()
    if d_infnorm != 0:
        d = d / d_infnorm
        d_sqnorm = numpy.sqrt(d @ d)
        if d_sqnorm != 0:
            d = d / d_sqnorm

    g_infnorm = numpy.abs(g).max()
    if g_infnorm != 0:
        g = g / g_infnorm
        g_sqnorm = numpy.sqrt(g @ g)
        if g_sqnorm != 0:
            g = g / g_sqnorm

    g = -g  # 改成下降方向

    if not d @ d:
        x = g.reshape(-1, 1)
    else:
        cos_gd: float = (g @ d) / numpy.sqrt((g @ g) * (d @ d))  # type: ignore
        if numpy.abs(cos_gd - 1) < numpy.sqrt(_eps):
            x = g.reshape(-1, 1)
        else:
            rad = numpy.linspace(0, numpy.arccos(cos_gd), num=100)
            w1, w2 = numpy.cos(rad), numpy.sin(rad)
            d = (d - g * w1[-1]) / w2[-1]  # 正交化
            x = w1 * g.reshape(-1, 1) + w2 * d.reshape(-1, 1)

    xx = clip_sol(x, orig_g, H.value, constraints, delta)
    return Status(xx, ret1.iter, ret1.flag, delta, orig_g, H.value)
