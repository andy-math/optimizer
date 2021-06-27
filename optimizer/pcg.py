# -*- coding: utf-8 -*-
from __future__ import annotations

import enum
import math
from typing import Optional, Tuple

import numpy
from numerical.linneq import check, constraint_check, margin
from numerical.typedefs import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float


@enum.unique
class PCG_Flag(enum.Enum):
    RESIDUAL_CONVERGENCE = enum.auto()
    NEGATIVE_CURVATURE = enum.auto()
    OUT_OF_TRUST_REGION = enum.auto()
    VIOLATE_CONSTRAINTS = enum.auto()


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

    for iter in range(n + 1):
        # 残差收敛性检查
        if numpy.max(numpy.abs(z)) < numpy.sqrt(_eps):
            return (p, direct, iter, PCG_Flag.RESIDUAL_CONVERGENCE)

        # 残差始终不收敛则是hessian矩阵病态，适用于非正定-负曲率情形
        if iter == n:
            return (p, direct, iter, PCG_Flag.NEGATIVE_CURVATURE)  # pragma: no cover

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
        if p is None:
            return None
        return float(g.T @ p + (0.5 * p).T @ H @ p)

    # 主循环
    p: ndarray
    direct: ndarray
    iter: int
    exit_flag: PCG_Flag
    p, direct, iter, exit_flag = _impl(g, H, constraints, delta)

    def make_valid_gradient(
        exit_flag: PCG_Flag,
    ) -> Tuple[Optional[ndarray], PCG_Flag]:
        assert iter == 0
        p = direct  # 使用二阶信息缩小变化太快的维度上的梯度
        norm_p = float(numpy.linalg.norm(p))  # type: ignore
        if norm_p > 0:
            p /= norm_p
        p = p * delta
        (n,) = p.shape
        lb, ub = margin(numpy.zeros((n,)), constraints)
        eliminated = numpy.zeros((n,), dtype=numpy.bool_)
        while True:
            index: numpy.ndarray
            index = numpy.logical_or(p < lb, ub < p)
            eliminated[index] = True
            p[eliminated] /= 2.0
            norm_p = float(numpy.linalg.norm(p))  # type: ignore
            if norm_p > 0:
                p /= norm_p
            p = p * delta
            if numpy.all(numpy.logical_and(lb <= p, p <= ub)):
                break
            if numpy.all(eliminated):
                return None, PCG_Flag.VIOLATE_CONSTRAINTS
        p.shape = (n, 1)
        if not check(p, constraints):
            return None, PCG_Flag.VIOLATE_CONSTRAINTS
        p.shape = (n,)
        if bool(numpy.any(eliminated)):
            return p, PCG_Flag.VIOLATE_CONSTRAINTS
        return p, exit_flag

    def make_valid_optimal(
        exit_flag: PCG_Flag,
    ) -> Tuple[ndarray, PCG_Flag]:
        assert exit_flag == PCG_Flag.OUT_OF_TRUST_REGION
        nonlocal direct, iter
        (n,) = p.shape
        norm_d = float(numpy.linalg.norm(direct))  # type: ignore
        if norm_d > 0:
            direct /= norm_d
        distance = float(numpy.sqrt(delta * delta - p.T @ p))  # 勾股定理
        p_new = p + distance * direct
        p_new.shape = (n, 1)
        if check(p_new, constraints):
            p_new.shape = (n,)
            iter += 1
            return p_new, exit_flag
        p_new.shape = (n,)
        return p, exit_flag

    # 残差收敛：对迭代成功和失败均适用
    if exit_flag == PCG_Flag.RESIDUAL_CONVERGENCE:
        return PCG_Status(p, fval(p), iter, exit_flag)

    # 负曲率：迭代成功时不再前进，迭代失败时返回裁剪梯度
    if exit_flag == PCG_Flag.NEGATIVE_CURVATURE:
        if iter > 0:
            return PCG_Status(p, fval(p), iter, exit_flag)  # pragma: no cover
        else:
            p_clip, exit_flag = make_valid_gradient(exit_flag)
            return PCG_Status(p_clip, fval(p_clip), iter, exit_flag)

    # 超出信赖域：迭代成功时前进，迭代失败时返回裁剪梯度
    if exit_flag == PCG_Flag.OUT_OF_TRUST_REGION:
        if iter > 0:
            p_clip, exit_flag = make_valid_optimal(exit_flag)
            return PCG_Status(p_clip, fval(p_clip), iter, exit_flag)
        else:
            p_clip, exit_flag = make_valid_gradient(exit_flag)
            return PCG_Status(p_clip, fval(p_clip), iter, exit_flag)

    # 违反约束：迭代成功时不再前进，迭代失败时返回裁剪梯度
    if exit_flag == PCG_Flag.VIOLATE_CONSTRAINTS:
        if iter > 0:
            return PCG_Status(p, fval(p), iter, exit_flag)  # pragma: no cover
        else:
            p_clip, exit_flag = make_valid_gradient(exit_flag)
            return PCG_Status(p_clip, fval(p_clip), iter, exit_flag)

    # 其它情形：不应存在
    assert False  # pragma: no cover
