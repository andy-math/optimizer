# -*- coding: utf-8 -*-
from typing import Callable, Optional

import numpy
from mypy_extensions import NamedArg
from numerical import findiff, linneq
from numerical.typedefs import ndarray

from optimizer import pcg


class Trust_Region_Options:
    tol_step: float = 1.0e-10
    tol_grad: float = 1.0e-6
    init_delta: float = 1.0
    max_iter: int
    format: Optional[
        Callable[
            [
                NamedArg(int, "iter"),
                NamedArg(float, "fval"),  # noqa: F821
                NamedArg(float, "step"),  # noqa: F821
                NamedArg(float, "grad"),  # noqa: F821
                NamedArg(float, "CGiter"),  # noqa: F821
                NamedArg(Optional[pcg.PCG_EXIT_FLAG], "CGexit"),  # noqa: F821
            ],
            str,
        ]
    ] = "iter = {iter: 5d}, fval = {fval: 13.6g}, step = {step: 13.6g}, grad = {grad: 12.3g}, CG = {CGiter: 7d}, {CGexit}".format  # noqa: E501

    def __init__(self, *, max_iter: int) -> None:
        self.max_iter = max_iter


class Trust_Region_Result:
    x: ndarray
    iter: int
    success: bool

    def __init__(self, x: ndarray, iter: int, *, success: bool) -> None:
        self.x = x
        self.iter = iter
        self.success = success


def trust_region(
    objective: Callable[[ndarray], float],
    gradient: Callable[[ndarray], ndarray],
    x: ndarray,
    opts: Trust_Region_Options,
) -> Trust_Region_Result:
    n = x.shape[0]
    constr_A = numpy.zeros((0, n))
    constr_b = numpy.zeros((0,))
    constr_lb = numpy.full((n,), -numpy.inf)
    constr_ub = numpy.full((n,), numpy.inf)

    assert opts.format is not None

    iter: int = 0
    pcg_iter: int = 0
    delta: float = opts.init_delta
    step_size: float = 0.0

    isposdef: Optional[bool] = None
    exit_flag: Optional[pcg.PCG_EXIT_FLAG] = None

    fval: float = objective(x)
    grad: ndarray = gradient(x)
    grad_infnorm: float = numpy.max(numpy.abs(grad))
    H = findiff.findiff(gradient, x, constr_A, constr_b, constr_lb, constr_ub)

    while True:

        print(
            opts.format(
                iter=iter,
                fval=fval,
                step=step_size,
                grad=grad_infnorm,
                CGiter=pcg_iter,
                CGexit=exit_flag,
            )
        )

        # 成功收敛准则
        if isposdef is not None and exit_flag is not None:  # PCG正定收敛
            if isposdef and exit_flag == pcg.PCG_EXIT_FLAG.RESIDUAL_CONVERGENCE:
                if grad_infnorm < opts.tol_grad:  # 梯度足够小
                    return Trust_Region_Result(x, iter, success=True)
                if step_size < opts.tol_step:  # 步长足够小
                    return Trust_Region_Result(x, iter, success=True)

        # 失败收敛准则
        if iter and step_size < opts.tol_step:  # 步长太小而不满足PCG正定收敛
            return Trust_Region_Result(x, iter, success=False)
        if iter > opts.max_iter:  # 迭代次数超过要求
            return Trust_Region_Result(x, iter, success=False)

        # PCG
        step: ndarray
        qpval: float
        step, qpval, isposdef, iter, exit_flag = pcg.pcg(grad, H, delta)
        step_size = numpy.linalg.norm(step)  # type: ignore
        iter += 1

        # 试探更新自变量
        new_x = x + step

        # 违反约束时丢弃isposdef和exit_flag以拒绝判定为收敛
        new_x.shape = (new_x.shape[0], 1)
        if not linneq.check(new_x, constr_A, constr_b, constr_lb, constr_ub):
            isposdef, exit_flag = None, None
            delta = step_size / 4
            continue
        new_x.shape = (new_x.shape[0],)

        # 对通过约束检查的自变量进行函数求值
        new_fval = objective(new_x)

        # 根据下降率确定信赖域缩放
        reduce: float = new_fval - fval
        ratio: float = 1 if reduce <= qpval else (0 if reduce >= 0 else reduce / qpval)
        if ratio >= 0.75 and step_size >= 0.9 * delta:
            delta *= 2
        elif ratio <= 0.25:
            delta = step_size / 4

        # 对符合下降要求的候选点进行更新
        if new_fval < fval:
            x, fval, grad = new_x, new_fval, gradient(new_x)
            grad_infnorm = numpy.max(numpy.abs(grad))
            H = findiff.findiff(gradient, x, constr_A, constr_b, constr_lb, constr_ub)
