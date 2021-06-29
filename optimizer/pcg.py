# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import numpy
from numpy import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float

from optimizer._internals.common.linneq import check, constraint_check
from optimizer._internals.pcg import flag, status
from optimizer._internals.pcg.policies import subspace_decay
from optimizer._internals.pcg.precondition import gradient_precon, hessian_precon
from optimizer._internals.trust_region.grad_maker import Hessian

Flag = flag.Flag
Status = status.Status


def _impl_input_check(
    input: Tuple[
        ndarray,
        ndarray,
        ndarray,
        Tuple[ndarray, ndarray, ndarray, ndarray],
        float,
    ]
) -> None:
    g, H, R, constraints, delta = input
    assertNoInfNaN(g)
    assertNoInfNaN(H)
    assertNoInfNaN(R)
    constraint_check(constraints)
    assertNoInfNaN_float(delta)


def _impl_output_check(output: Tuple[Status, Optional[ndarray]]) -> None:
    status, direct = output
    if status.flag == Flag.RESIDUAL_CONVERGENCE:
        assert direct is None
    else:
        assert direct is not None
        assertNoInfNaN(direct)


@bind_checker.bind_checker_5(input=_impl_input_check, output=_impl_output_check)
def _implimentation(
    g: ndarray,
    H: ndarray,
    R: ndarray,
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

    _eps = float(numpy.finfo(numpy.float64).eps)

    (n,) = g.shape
    x: ndarray = numpy.zeros((n,))  # 目标点
    r: ndarray = -g  # 残差
    z: ndarray = r / R  # 归一化后的残差
    d: ndarray = z  # 搜索方向

    inner1: float = float(r.T @ z)

    for iter in range(n):
        # 残差收敛性检查
        if numpy.max(numpy.abs(z)) < numpy.sqrt(_eps):
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
        if numpy.linalg.norm(x_new) > delta:  # type: ignore
            return exit_(x, d, iter, Flag.OUT_OF_TRUST_REGION)

        # 违反约束
        x_new.shape = (n, 1)
        if not check(x_new, constraints):
            return exit_(x, d, iter, Flag.VIOLATE_CONSTRAINTS)
        x_new.shape = (n,)

        # 更新坐标点
        x = x_new

        # 更新残差
        r = r - alpha * ww
        z = r / R

        # 更新搜索方向
        inner2: float = inner1
        inner1 = float(r.T @ z)
        beta: float = inner1 / inner2
        d = z + beta * d

    return exit_(x, None, iter, Flag.RESIDUAL_CONVERGENCE)


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

    return status.best_status(
        _best_policy(g, H.value, hessian_precon(H.value), constraints, delta),
        _best_policy(g, H.value, gradient_precon(g), constraints, delta),
        subspace_decay(
            g,
            H.value,
            Status(None, 0, Flag.POLICY_ONLY, delta, g, H.value),
            -H.pinv @ g,
            delta,
            constraints,
        ),
    )


def _best_policy(
    g: ndarray,
    H: ndarray,
    R: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Status:

    ret0 = subspace_decay(
        g,
        H,
        Status(None, 0, Flag.POLICY_ONLY, delta, g, H),
        -g / R,
        delta,
        constraints,
    )
    ret1, direct = _implimentation(g, H, R, constraints, delta)
    if ret1.flag == Flag.RESIDUAL_CONVERGENCE:
        assert direct is None
        ret2 = None
    else:
        assert direct is not None
        ret2 = subspace_decay(g, H, ret1, direct, delta, constraints)
    return status.best_status(ret0, ret1, ret2)
