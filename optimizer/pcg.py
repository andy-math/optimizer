# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy
from numerical.linneq import check, constraint_check
from numerical.typedefs import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float

from optimizer._internals.pcg import flags
from optimizer._internals.pcg.policies import subspace_decay

PCG_Flag = flags.PCG_Flag


class PCG_Status:
    x: Optional[ndarray]
    fval: Optional[float]
    iter: int
    flag: PCG_Flag
    size: Optional[float]

    def __init__(
        self,
        x: Optional[ndarray],
        fval: Optional[float],
        iter: int,
        flag: PCG_Flag,
    ) -> None:
        self.x = x
        self.fval = fval
        self.iter = iter
        self.flag = flag
        self.size = None if x is None else math.sqrt(float(x @ x))


def _input_check(
    input: Tuple[ndarray, ndarray, Tuple[ndarray, ndarray, ndarray, ndarray], float]
) -> None:
    g, H, constraints, delta = input
    assertNoInfNaN(g)
    assertNoInfNaN(H)
    constraint_check(constraints)
    assertNoInfNaN_float(delta)


def _impl_output_check(output: Tuple[ndarray, ndarray, int, PCG_Flag]) -> None:
    p, direct, _, _ = output
    assertNoInfNaN(p)
    assertNoInfNaN(direct)


N = dyn_typing.SizeVar()
nConstraints = dyn_typing.SizeVar()


@dyn_typing.dyn_check_4(
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
    output=dyn_typing.Tuple(
        (
            dyn_typing.NDArray(numpy.float64, (N,)),
            dyn_typing.NDArray(numpy.float64, (N,)),
            dyn_typing.Int(),
            dyn_typing.Class(PCG_Flag),
        )
    ),
)
@bind_checker.bind_checker_4(input=_input_check, output=_impl_output_check)
def _impl(
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Tuple[ndarray, ndarray, int, PCG_Flag]:

    # 取 max{ l2norm(col(H)), sqrt(eps) }
    # 预条件子 M = C.T @ C == diag(R)
    # 其中 H === H.T  =>  norm(col(H)) === norm(row(H))
    _eps = float(numpy.finfo(numpy.float64).eps)
    dnrms: ndarray = numpy.sqrt(numpy.sum(H * H, axis=1))
    R: ndarray = numpy.maximum(dnrms, numpy.sqrt(numpy.array([_eps])))

    (n,) = g.shape
    p: ndarray = numpy.zeros((n,))  # 目标点
    r: ndarray = -g  # 残差
    z: ndarray = r / R  # 归一化后的残差
    direct: ndarray = z  # 搜索方向

    inner1: float = float(r.T @ z)

    for iter in range(n):
        # 残差收敛性检查
        if numpy.max(numpy.abs(z)) < numpy.sqrt(_eps):
            return (p, direct, iter, PCG_Flag.RESIDUAL_CONVERGENCE)

        # 负曲率检查
        ww: ndarray = H @ direct
        denom: float = float(direct.T @ ww)
        if denom <= 0:
            return (p, direct, iter, PCG_Flag.NEGATIVE_CURVATURE)

        # 试探坐标点
        alpha: float = inner1 / denom
        pnew: ndarray = p + alpha * direct

        # 目标点超出信赖域
        if numpy.linalg.norm(pnew) > delta:  # type: ignore
            return (p, direct, iter, PCG_Flag.OUT_OF_TRUST_REGION)

        # 违反约束
        pnew.shape = (n, 1)
        if not check(pnew, constraints):
            return (p, direct, iter, PCG_Flag.VIOLATE_CONSTRAINTS)  # pragma: no cover
        pnew.shape = (n,)

        # 更新坐标点
        p = pnew

        # 更新残差
        r = r - alpha * ww
        z = r / R

        # 更新搜索方向
        inner2: float = inner1
        inner1 = float(r.T @ z)
        beta: float = inner1 / inner2
        direct = z + beta * direct

    if numpy.max(numpy.abs(z)) < numpy.sqrt(_eps):
        return (p, direct, iter, PCG_Flag.RESIDUAL_CONVERGENCE)

    assert False  # pragma: no cover


def _pcg_output_check(output: PCG_Status) -> None:
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
    output=dyn_typing.Class(PCG_Status),
)
@bind_checker.bind_checker_4(input=_input_check, output=_pcg_output_check)
def pcg(
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> PCG_Status:
    def fval(p: Optional[ndarray]) -> Optional[float]:
        return None if p is None else float(g.T @ p + (0.5 * p).T @ H @ p)

    # 主循环
    p: Optional[ndarray]
    p, direct, iter, exit_flag = _impl(g, H, constraints, delta)
    if exit_flag != PCG_Flag.RESIDUAL_CONVERGENCE:
        p, exit_flag = subspace_decay(p, direct, delta, constraints, exit_flag)
    return PCG_Status(p, fval(p), iter, exit_flag)
