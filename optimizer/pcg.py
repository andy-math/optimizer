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
from optimizer._internals.pcg.precondition import gradient_precon, hessian_precon

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


def _impl_input_check(
    input: Tuple[
        ndarray, ndarray, ndarray, Tuple[ndarray, ndarray, ndarray, ndarray], float
    ]
) -> None:
    g, H, R, constraints, delta = input
    assertNoInfNaN(g)
    assertNoInfNaN(H)
    assertNoInfNaN(R)
    constraint_check(constraints)
    assertNoInfNaN_float(delta)


def _impl_output_check(
    output: Tuple[ndarray, Optional[ndarray], int, PCG_Flag]
) -> None:
    p, direct, _, _ = output
    assertNoInfNaN(p)
    if direct is not None:
        assertNoInfNaN(direct)


N = dyn_typing.SizeVar()
nConstraints = dyn_typing.SizeVar()


@dyn_typing.dyn_check_5(
    input=(
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.NDArray(numpy.float64, (N, N)),
        dyn_typing.NDArray(numpy.float64, (N,)),
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
            dyn_typing.Optional(dyn_typing.NDArray(numpy.float64, (N,))),
            dyn_typing.Int(),
            dyn_typing.Class(PCG_Flag),
        )
    ),
)
@bind_checker.bind_checker_5(input=_impl_input_check, output=_impl_output_check)
def _implimentation(
    g: ndarray,
    H: ndarray,
    R: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> Tuple[ndarray, Optional[ndarray], int, PCG_Flag]:
    _eps = float(numpy.finfo(numpy.float64).eps)

    (n,) = g.shape
    p: ndarray = numpy.zeros((n,))  # 目标点
    r: ndarray = -g  # 残差
    z: ndarray = r / R  # 归一化后的残差
    direct: ndarray = z  # 搜索方向

    inner1: float = float(r.T @ z)

    for iter in range(n):
        # 残差收敛性检查
        if numpy.max(numpy.abs(z)) < numpy.sqrt(_eps):
            return (p, None, iter, PCG_Flag.RESIDUAL_CONVERGENCE)

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

    return (p, None, iter, PCG_Flag.RESIDUAL_CONVERGENCE)


def _pcg_input_check(
    input: Tuple[ndarray, ndarray, Tuple[ndarray, ndarray, ndarray, ndarray], float]
) -> None:
    g, H, constraints, delta = input
    assertNoInfNaN(g)
    assertNoInfNaN(H)
    constraint_check(constraints)
    assertNoInfNaN_float(delta)


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
@bind_checker.bind_checker_4(input=_pcg_input_check, output=_pcg_output_check)
def pcg(
    g: ndarray,
    H: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> PCG_Status:

    ret1 = _best_policy(g, H, hessian_precon(H), constraints, delta)
    ret2 = _best_policy(g, H, gradient_precon(g), constraints, delta)

    return _best_status(ret1, ret2)


def _best_status(ret1: PCG_Status, ret2: PCG_Status) -> PCG_Status:
    def norm2(x: ndarray) -> float:
        return math.sqrt(float(x @ x))

    if ret1.x is None and ret2.x is None:
        return ret1
    elif ret1.x is None:
        return ret2
    elif ret2.x is None:
        return ret1
    else:
        assert ret1.fval is not None
        assert ret2.fval is not None
        if ret1.fval < ret2.fval or (
            ret1.fval == ret2.fval and norm2(ret1.x) <= norm2(ret2.x)
        ):
            return ret1
        else:
            return ret2


def _best_policy(
    g: ndarray,
    H: ndarray,
    R: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    delta: float,
) -> PCG_Status:
    def fval(p: Optional[ndarray]) -> Optional[float]:
        return None if p is None else float(g.T @ p + (0.5 * p).T @ H @ p)

    _p0, _exit0 = subspace_decay(
        g, H, numpy.zeros(g.shape), -g / R, delta, constraints, PCG_Flag.POLICY_ONLY
    )
    ret0 = PCG_Status(_p0, fval(_p0), 0, _exit0)

    _p1, _direct, _iter, _exit1 = _implimentation(g, H, R, constraints, delta)
    ret1 = PCG_Status(_p1, fval(_p1), _iter, _exit1)

    if _exit1 == PCG_Flag.RESIDUAL_CONVERGENCE:
        assert _direct is None
    else:
        assert _direct is not None
        _p2, _exit2 = subspace_decay(g, H, _p1, _direct, delta, constraints, _exit1)
        ret2 = PCG_Status(_p2, fval(_p2), _iter, _exit2)
        if _iter:
            ret1 = _best_status(ret1, ret2)
        else:
            ret1 = ret2
    return _best_status(ret1, ret0)
