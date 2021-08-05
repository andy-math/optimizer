# -*- coding: utf-8 -*-


from typing import Optional, Tuple, cast

import numpy

from optimizer._internals.common.linneq import constraint_check
from optimizer._internals.common.norm import norm_l2, safe_normalize
from optimizer._internals.pcg import flag, status
from optimizer._internals.pcg.circular_interp import circular_interp
from optimizer._internals.pcg.clip_solution import clip_solution
from optimizer._internals.pcg.qpval import QuadEvaluator
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float
from overloads.typedefs import ndarray

Flag = flag.Flag
Status = status.Status
_eps = float(numpy.finfo(numpy.float64).eps)


N = dyn_typing.SizeVar()
nConstraints = dyn_typing.SizeVar()

"""
动态类型签名
"""
dyn_signature = dyn_typing.dyn_check_3(
    input=(
        dyn_typing.Class(QuadEvaluator),
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


def no_check(_: QuadEvaluator) -> None:
    pass


def _impl_output_check(output: Tuple[Status, Optional[ndarray]]) -> None:
    status, direct = output
    if status.flag == Flag.RESIDUAL_CONVERGENCE:
        assert direct is None
    else:
        assert direct is not None
        assertNoInfNaN(direct)


@bind_checker.bind_checker_3(
    input=bind_checker.make_checker_3(no_check, constraint_check, assertNoInfNaN_float),
    output=_impl_output_check,
)
def _implimentation(
    qpval: QuadEvaluator,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Tuple[Status, Optional[ndarray]]:
    def exit_(
        x: ndarray, d: Optional[ndarray], iter: int, flag: Flag
    ) -> Tuple[Status, Optional[ndarray]]:
        return Status(x, iter, flag, delta, qpval), d

    g, H = qpval.g, qpval.H

    # 归一化初始残差以防止久不收敛
    R: float = max(norm_l2(g), numpy.sqrt(_eps))

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
@bind_checker.bind_checker_3(
    input=bind_checker.make_checker_3(no_check, constraint_check, assertNoInfNaN_float),
    output=_pcg_output_check,
)
def pcg(
    qpval: QuadEvaluator,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Status:
    g, H = qpval.g, qpval.H
    status, direct = _implimentation(qpval, constraints, delta)
    d = status.x
    if direct is not None:
        assert status.flag != Flag.RESIDUAL_CONVERGENCE
        d = d + clip_direction(direct, g, H, constraints, delta, basement=d)
    x = circular_interp(-g, d)
    x_clip = clip_solution(x, g, H, constraints, delta)
    x_g = clip_direction(-g, g, H, constraints, delta)
    x_d = clip_direction(d, g, H, constraints, delta)
    x_lstsq = clip_direction(
        numpy.linalg.lstsq(H, -g)[0],  # type: ignore
        g,
        H,
        constraints,
        delta,
    )
    assert qpval(x_clip) <= qpval(x_g) + 1e-6
    assert qpval(x_clip) <= qpval(x_d) + 1e-6
    if qpval(x_clip) <= qpval(x_lstsq):
        return Status(x_clip, status.iter, status.flag, delta, qpval)
    else:
        return Status(x_lstsq, status.iter, status.flag, delta, qpval)
