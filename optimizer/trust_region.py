# -*- coding: utf-8 -*-
from typing import Callable, Optional, Tuple

import numpy
from mypy_extensions import NamedArg
from numerical import difference, findiff, linneq
from numerical.isposdef import isposdef
from numerical.typedefs import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN

from optimizer import pcg


class Grad_Check_Failed(BaseException):
    checker: Optional[Callable[[ndarray, ndarray], float]]
    analytic: ndarray
    findiff_: ndarray

    def __init__(
        self,
        checker: Callable[[ndarray, ndarray], float],
        analytic: ndarray,
        findiff_: ndarray,
    ) -> None:
        self.checker = checker
        self.analytic = analytic
        self.findiff_ = findiff_


class Trust_Region_Options:
    tol_step: float = 1.0e-10
    tol_grad: float = 1.0e-6
    init_delta: float = 1.0
    max_iter: int
    check_rel: float = 1.0e-2
    check_abs: Optional[float] = None
    check_iter: Optional[int] = None
    format: Optional[
        Callable[
            [
                NamedArg(int, "iter"),
                NamedArg(float, "fval"),  # noqa: F821
                NamedArg(float, "step"),  # noqa: F821
                NamedArg(float, "grad"),  # noqa: F821
                NamedArg(float, "CGiter"),  # noqa: F821
                NamedArg(str, "CGexit"),  # noqa: F821
                NamedArg(str, "posdef"),  # noqa: F821
            ],
            str,
        ]
    ]
    CGexit: Optional[Callable[[Optional[pcg.PCG_EXIT_FLAG]], str]]
    posdef: Optional[Callable[[ndarray], str]]

    def __init__(self, *, max_iter: int) -> None:
        self.max_iter = max_iter
        self.format = "iter = {iter: 5d}, fval = {fval: 13.6g}, step = {step: 13.6g}, grad = {grad: 12.3g}, CG = {CGiter: 7d}, {CGexit} {posdef}".format  # noqa: E501
        self.CGexit = lambda x: x.name if x is not None else "None"
        self.posdef = lambda H: "-*- ill -*-" if not isposdef(H) else ""


class Trust_Region_Result:
    x: ndarray
    iter: int
    gradient: ndarray
    success: bool

    def __init__(self, x: ndarray, iter: int, grad: ndarray, *, success: bool) -> None:
        self.x = x
        self.iter = iter
        self.gradient = grad
        self.success = success


def _input_check(
    input: Tuple[
        Callable[[ndarray], float],
        Callable[[ndarray], ndarray],
        ndarray,
        ndarray,
        ndarray,
        ndarray,
        ndarray,
        Trust_Region_Options,
    ]
) -> None:
    _, _, x, A, b, lb, ub, _ = input
    linneq.constraint_check(A, b, lb, ub, theta=x)


def _output_check(output: Trust_Region_Result) -> None:
    assertNoInfNaN(output.x)


N = dyn_typing.SizeVar()
nConstraint = dyn_typing.SizeVar()


@dyn_typing.dyn_check_8(
    input=(
        dyn_typing.Callable(),
        dyn_typing.Callable(),
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.NDArray(numpy.float64, (nConstraint, N)),
        dyn_typing.NDArray(numpy.float64, (nConstraint,)),
        dyn_typing.NDArray(numpy.float64, (N,)),  # force line wrap
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.Class(Trust_Region_Options),
    ),
    output=dyn_typing.Class(Trust_Region_Result),
)
@bind_checker.bind_checker_8(input=_input_check, output=_output_check)
def trust_region(
    objective: Callable[[ndarray], float],
    gradient: Callable[[ndarray], ndarray],
    x: ndarray,
    constr_A: ndarray,
    constr_b: ndarray,
    constr_lb: ndarray,
    constr_ub: ndarray,
    opts: Trust_Region_Options,
) -> Trust_Region_Result:
    def objective_ndarray(x: ndarray) -> ndarray:
        return numpy.array([objective(x)])

    iter: int = 0
    grad_infnorm: float = numpy.inf
    init_grad_infnorm: float = 0

    def make_grad(x: ndarray) -> ndarray:
        analytic = gradient(x)
        if opts.check_abs is None:
            if opts.check_iter is not None and iter > opts.check_iter:
                return analytic
            if grad_infnorm < init_grad_infnorm * opts.check_rel:
                return analytic

        findiff_ = findiff.findiff(
            objective_ndarray, x, constr_A, constr_b, constr_lb, constr_ub
        )
        assert len(findiff_.shape) == 2 and findiff_.shape[0] == 1
        findiff_.shape = (findiff_.shape[1],)

        if difference.relative(analytic, findiff_) > opts.check_rel:
            raise Grad_Check_Failed(difference.relative, analytic, findiff_)
        if opts.check_abs is not None:
            if difference.absolute(analytic, findiff_) > opts.check_abs:
                raise Grad_Check_Failed(difference.absolute, analytic, findiff_)
        return analytic

    assert opts.CGexit is not None
    assert opts.posdef is not None

    pcg_iter: int = 0
    delta: float = opts.init_delta
    step_size: float = 0.0

    exit_flag: Optional[pcg.PCG_EXIT_FLAG] = None

    assert linneq.check(x.reshape(-1, 1), constr_A, constr_b, constr_lb, constr_ub)
    fval: float = objective(x)
    grad: ndarray = make_grad(x)
    grad_infnorm = numpy.max(numpy.abs(grad))
    init_grad_infnorm = grad_infnorm
    H = findiff.findiff(gradient, x, constr_A, constr_b, constr_lb, constr_ub)
    H = (H.T + H) / 2
    constraints = constr_A, constr_b - constr_A @ x, constr_lb - x, constr_ub - x

    while True:
        if opts.format is not None:
            print(
                opts.format(
                    iter=iter,
                    fval=fval,
                    step=step_size,
                    grad=grad_infnorm,
                    CGiter=pcg_iter,
                    CGexit=opts.CGexit(exit_flag),
                    posdef=opts.posdef(H),
                )
            )

        # PCG
        step: Optional[ndarray]
        qpval: Optional[float]
        step, qpval, pcg_iter, exit_flag = pcg.pcg(grad, H, constraints, delta)
        iter += 1

        # 成功收敛准则
        assert exit_flag is not None
        if exit_flag == pcg.PCG_EXIT_FLAG.RESIDUAL_CONVERGENCE:  # PCG正定收敛
            if grad_infnorm < opts.tol_grad:  # 梯度足够小
                return Trust_Region_Result(x, iter, grad, success=True)
            if step_size < opts.tol_step:  # 步长足够小
                return Trust_Region_Result(
                    x, iter, grad, success=True
                )  # pragma: no cover

        # 失败收敛准则
        if delta < opts.tol_step:  # 步长太小
            return Trust_Region_Result(x, iter, grad, success=False)  # pragma: no cover
        if iter > opts.max_iter:  # 迭代次数超过要求
            return Trust_Region_Result(x, iter, grad, success=False)  # pragma: no cover

        if step is None:
            delta = step_size / 4.0
            continue

        assert qpval is not None

        step_size = numpy.linalg.norm(step)  # type: ignore

        # 试探更新自变量
        new_x = x + step

        # 对通过约束检查的自变量进行函数求值
        new_fval = objective(new_x)

        # 根据下降率确定信赖域缩放
        reduce: float = new_fval - fval
        ratio: float = 1 if reduce <= qpval else (0 if reduce >= 0 else reduce / qpval)
        if ratio >= 0.75 and step_size >= 0.9 * delta:
            delta *= 2
        elif ratio <= 0.25:
            delta = step_size / 4.0

        # 对符合下降要求的候选点进行更新
        if new_fval < fval:
            x, fval, grad = new_x, new_fval, make_grad(new_x)
            grad_infnorm = numpy.max(numpy.abs(grad))
            H = findiff.findiff(gradient, x, constr_A, constr_b, constr_lb, constr_ub)
            H = (H.T + H) / 2
            constraints = (
                constr_A,
                constr_b - constr_A @ x,
                constr_lb - x,
                constr_ub - x,
            )
