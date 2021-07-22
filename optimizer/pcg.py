# -*- coding: utf-8 -*-


from typing import Optional, Tuple, cast

import numpy
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float
from overloads.typing import ndarray

from optimizer._internals.common.linneq import constraint_check
from optimizer._internals.common.norm import norm_l2, safe_normalize
from optimizer._internals.pcg import flag, status
from optimizer._internals.pcg.circular_interp import circular_interp
from optimizer._internals.pcg.clip_solution import clip_solution
from optimizer._internals.pcg.qpval import qpval

Flag = flag.Flag
Status = status.Status
_eps = float(numpy.finfo(numpy.float64).eps)


N = dyn_typing.SizeVar()
nConstraints = dyn_typing.SizeVar()

"""
动态类型签名
"""
dyn_signature = dyn_typing.dyn_check_4(
    input=(
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.NDArray(numpy.float64, (N, N)),
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


def _pcg_input_check(
    input: Tuple[ndarray, ndarray, Tuple[ndarray, ndarray, ndarray, ndarray], float]
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


@bind_checker.bind_checker_4(input=_pcg_input_check, output=_impl_output_check)
def _implimentation(
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Tuple[Status, Optional[ndarray]]:
    def exit_(
        x: ndarray, d: Optional[ndarray], iter: int, flag: Flag
    ) -> Tuple[Status, Optional[ndarray]]:
        return Status(x, iter, flag, delta, g, H), d

    assert numpy.all(H.T == H)

    # 取 max{ l2norm(col(H)), sqrt(eps) }
    # 预条件子 M = C.T @ C == diag(R)
    # 其中 H === H.T  =>  norm(col(H)) === norm(row(H))
    R: ndarray
    H_max = numpy.abs(H).max(axis=1, keepdims=True)
    assert H_max.shape[1] == 1
    R = H_max[:, 0] * numpy.sqrt(numpy.sum((H / H_max) * (H / H_max), axis=1))
    R = numpy.minimum(R, norm_l2(g))
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


def clip_direction(
    x: ndarray,
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
    *,
    basement: Optional[ndarray] = None
) -> ndarray:
    A, b, lb, ub = constraints
    if basement is not None:
        delta = numpy.sqrt(delta * delta - basement @ basement)
        assert numpy.isfinite(delta)
        g = g + H @ basement
        b = b - A @ basement
        lb = lb - basement
        ub = ub - basement
    x = safe_normalize(x).reshape((-1, 1))
    return clip_solution(x, g, H, (A, b, lb, ub), delta)


def _pcg_output_check(output: Status) -> None:
    pass


@dyn_signature
@bind_checker.bind_checker_4(input=_pcg_input_check, output=_pcg_output_check)
def pcg(
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Status:
    status, direct = _implimentation(g, H, constraints, delta)
    d = status.x
    if direct is not None:
        assert status.flag != Flag.RESIDUAL_CONVERGENCE
        d = d + clip_direction(direct, g, H, constraints, delta, basement=d)
    x = circular_interp(-g, d)
    x_clip = clip_solution(x, g, H, constraints, delta)
    x_g = clip_direction(-g, g, H, constraints, delta)
    x_d = clip_direction(d, g, H, constraints, delta)
    assert qpval(g=g, H=H, x=x_clip) <= qpval(g=g, H=H, x=x_g)
    assert qpval(g=g, H=H, x=x_clip) <= qpval(g=g, H=H, x=x_d)
    return Status(x_clip, status.iter, status.flag, delta, g, H)
