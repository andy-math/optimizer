# -*- coding: utf-8 -*-
import enum
from typing import Tuple

import numpy
from numerical.typedefs import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN, assertNoInfNaN_float


@enum.unique
class PCG_EXIT_FLAG(enum.Enum):
    RESIDUAL_CONVERGENCE = enum.auto()
    NEGATIVE_CURVATURE = enum.auto()
    VIOLATE_CONSTRAINTS = enum.auto()
    OUT_OF_TRUST_REGION = enum.auto()
    NOT_CONVERGENCE = enum.auto()


def _input_check(input: Tuple[ndarray, ndarray, float]) -> None:
    g, H, delta = input
    assertNoInfNaN(g)
    assertNoInfNaN(H)
    assertNoInfNaN_float(delta)


def _impl_output_check(output: Tuple[ndarray, ndarray, int, PCG_EXIT_FLAG]) -> None:
    p, direct, _, _ = output
    assertNoInfNaN(p)
    assertNoInfNaN(direct)


N = dyn_typing.SizeVar()


@dyn_typing.dyn_check_3(
    input=(
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.NDArray(numpy.float64, (N, N)),
        dyn_typing.Float(),
    ),
    output=dyn_typing.Tuple(
        (
            dyn_typing.NDArray(numpy.float64, (N,)),
            dyn_typing.NDArray(numpy.float64, (N,)),
            dyn_typing.Int(),
            dyn_typing.Class(PCG_EXIT_FLAG),
        )
    ),
)
@bind_checker.bind_checker_3(input=_input_check, output=_impl_output_check)
def _impl(
    g: ndarray, H: ndarray, delta: float
) -> Tuple[ndarray, ndarray, int, PCG_EXIT_FLAG]:
    _eps = float(numpy.finfo(numpy.float64).eps)
    n: int = g.shape[0]
    dnrms: ndarray = numpy.sum(H * H, axis=1)
    R2: ndarray = numpy.maximum(dnrms, numpy.sqrt(numpy.array([_eps])))

    p: ndarray = numpy.zeros((n,))  # 目标点
    r: ndarray = -g  # 残差
    z: ndarray = r / R2  # 归一化后的残差
    direct: ndarray = z  # 搜索方向

    inner1: float = float(r.T @ z)

    for iter in range(n + 1):
        if numpy.max(numpy.abs(z)) < numpy.sqrt(_eps):
            return (
                p,
                direct,
                iter,
                PCG_EXIT_FLAG.RESIDUAL_CONVERGENCE,
            )
        if iter == n:
            return (
                p,
                direct,
                iter,
                PCG_EXIT_FLAG.NOT_CONVERGENCE,
            )
        ww: ndarray = H @ direct
        denom: float = float(direct.T @ ww)
        if denom <= 0:
            return (
                p,
                direct,
                iter,
                PCG_EXIT_FLAG.NEGATIVE_CURVATURE,
            )
        # 更新坐标点
        alpha: float = inner1 / denom
        pnew: ndarray = p + alpha * direct
        # 线性约束 TODO
        if False:
            return (
                p,
                direct,
                iter,
                PCG_EXIT_FLAG.VIOLATE_CONSTRAINTS,
            )
        # 信赖域
        if numpy.linalg.norm(pnew) > delta:  # type: ignore
            return (
                p,
                direct,
                iter,
                PCG_EXIT_FLAG.OUT_OF_TRUST_REGION,
            )
        p = pnew
        # 更新残差
        r = r - alpha * ww
        z = r / R2
        # 更新搜索方向
        inner2: float = inner1
        inner1 = float(r.T @ z)
        beta: float = inner1 / inner2
        direct = z + beta * direct
    assert False


def _pcg_output_check(output: Tuple[ndarray, float, bool, bool, int]) -> None:
    p, qpval, _, _, _ = output
    assertNoInfNaN(p)
    assertNoInfNaN_float(qpval)


@dyn_typing.dyn_check_3(
    input=(
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.NDArray(numpy.float64, (N, N)),
        dyn_typing.Float(),
    ),
    output=dyn_typing.Tuple(
        (
            dyn_typing.NDArray(numpy.float64, (N,)),
            dyn_typing.Float(),
            dyn_typing.Bool(),
            dyn_typing.Bool(),
            dyn_typing.Int(),
        )
    ),
)
@bind_checker.bind_checker_3(input=_input_check, output=_pcg_output_check)
def pcg(g: ndarray, H: ndarray, delta: float) -> Tuple[ndarray, float, bool, bool, int]:
    p: ndarray
    direct: ndarray
    iter: int
    exit_code: PCG_EXIT_FLAG
    p, direct, iter, exit_code = _impl(g, H, delta)
    posdef: bool
    wellDefined: bool
    if exit_code == PCG_EXIT_FLAG.RESIDUAL_CONVERGENCE:
        posdef, wellDefined = True, True

    elif exit_code == PCG_EXIT_FLAG.NEGATIVE_CURVATURE:
        posdef, wellDefined = False, True
        if not iter:  # CG迭代失败时按照一阶逼近取梯度
            norm_g: float = float(numpy.linalg.norm(g))  # type: ignore
            p = -g
            if norm_g > 0:
                p = p / norm_g
            p = delta * p

    elif exit_code == PCG_EXIT_FLAG.VIOLATE_CONSTRAINTS:
        posdef, wellDefined = True, False

    elif exit_code == PCG_EXIT_FLAG.OUT_OF_TRUST_REGION:
        posdef, wellDefined = True, True
        """
        pnew == p + alpha * d
        p.T @ p + alpha * alpha * d.T @ d == delta * delta
            (where p.T @ d === 0, due to orthogonality)
        alpha = sqrt( (delta * delta - p.T @ p)/(d.T @ d) )
        """
        alpha: float = float(
            numpy.sqrt((delta * delta - p.T @ p) / (direct.T @ direct))
        )
        p = p + alpha * direct
    elif exit_code == PCG_EXIT_FLAG.NOT_CONVERGENCE:
        posdef, wellDefined = True, False
    else:
        assert False
    qpval: float = float(g.T @ p + (0.5 * p).T @ H @ p)
    return p, qpval, posdef, wellDefined, iter
