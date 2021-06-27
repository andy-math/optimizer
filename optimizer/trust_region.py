# -*- coding: utf-8 -*-
from typing import Callable, NamedTuple, Optional, Tuple

import numpy
from numerical import linneq
from numerical.typedefs import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN

from optimizer import pcg
from optimizer._internals.trust_region import format, options
from optimizer._internals.trust_region.grad_maker import (
    Gradient,
    GradientCheck,
    make_gradient,
    make_hessian,
)

Trust_Region_Format_T = format.Trust_Region_Format_T
default_format = format.default_format
Trust_Region_Options = options.Trust_Region_Options


class Trust_Region_Result(NamedTuple):
    x: ndarray
    iter: int
    delta: float
    gradient: Gradient
    success: bool


def _input_check(
    input: Tuple[
        Callable[[ndarray], float],
        Callable[[ndarray], ndarray],
        ndarray,
        Tuple[
            ndarray,
            ndarray,
            ndarray,
            ndarray,
        ],
        Trust_Region_Options,
    ]
) -> None:
    _, _, x, constraints, _ = input
    linneq.constraint_check(constraints, theta=x)


def _output_check(output: Trust_Region_Result) -> None:
    assertNoInfNaN(output.x)


N = dyn_typing.SizeVar()
nConstraint = dyn_typing.SizeVar()


@dyn_typing.dyn_check_5(
    input=(
        dyn_typing.Callable(),
        dyn_typing.Callable(),
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.Tuple(
            (
                dyn_typing.NDArray(numpy.float64, (nConstraint, N)),
                dyn_typing.NDArray(numpy.float64, (nConstraint,)),
                dyn_typing.NDArray(numpy.float64, (N,)),  # force line wrap
                dyn_typing.NDArray(numpy.float64, (N,)),
            )
        ),
        dyn_typing.Class(Trust_Region_Options),
    ),
    output=dyn_typing.Class(Trust_Region_Result),
)
@bind_checker.bind_checker_5(input=_input_check, output=_output_check)
def trust_region(
    objective: Callable[[ndarray], float],
    gradient: Callable[[ndarray], ndarray],
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    opts: Trust_Region_Options,
) -> Trust_Region_Result:

    _hess_is_up_to_date: bool = False
    _hess_shaked: bool = False
    shaking: int = 0

    def objective_ndarray(x: ndarray) -> ndarray:
        return numpy.array([objective(x)])

    def output(
        iter: int,
        fval: float,
        step_size: float,
        grad_infnorm: float,
        CGiter: Optional[int],
        CGexit: Optional[pcg.PCG_EXIT_FLAG],
        hessian: ndarray,
    ) -> None:
        nonlocal _hess_shaked
        if opts.format is not None:
            output = opts.format(
                iter=iter,
                fval=fval,
                step=step_size,
                grad=grad_infnorm,
                CGiter=CGiter if CGiter is not None else 0,
                CGexit=CGexit.name if CGexit is not None else "None",
                posdef=opts.posdef(hessian) if opts.posdef is not None else "",
                shaking="Shaking" if _hess_shaked else "       ",
            )
            if output is not None:
                print(output)
        _hess_shaked = False

    def get_info(
        x: ndarray, iter: int, grad_infnorm: float, init_grad_infnorm: float
    ) -> Tuple[Gradient, Tuple[ndarray, ndarray, ndarray, ndarray]]:
        new_grad = make_gradient(
            gradient,
            x,
            constraints,
            opts,
            check=GradientCheck(
                objective_ndarray, iter, grad_infnorm, init_grad_infnorm
            ),
        )
        A, b, lb, ub = constraints
        _constraints = (A, b - A @ x, lb - x, ub - x)
        return new_grad, _constraints

    def make_hess(x: ndarray) -> ndarray:
        nonlocal _hess_is_up_to_date, shaking, _hess_shaked
        assert not _hess_is_up_to_date
        H = make_hessian(gradient, x, constraints, opts)
        _hess_is_up_to_date, _hess_shaked = True, True
        shaking = x.shape[0] if opts.shaking == "x.shape[0]" else opts.shaking
        return H

    iter: int = 0
    delta: float = opts.init_delta
    assert linneq.check(x.reshape(-1, 1), constraints)

    fval: float
    grad: Gradient
    H: ndarray
    _constr_shifted: Tuple[ndarray, ndarray, ndarray, ndarray]

    fval = objective(x)
    grad, _constr_shifted = get_info(x, iter, numpy.inf, 0.0)
    H = make_hess(x)
    output(iter, fval, numpy.nan, grad.infnorm, None, None, H)

    init_grad_infnorm = grad.infnorm
    old_fval, stall_iter = fval, 0
    while True:
        # 失败情形的截止条件放在最前是因为pcg失败时的continue会导致后面代码被跳过
        if delta < opts.tol_step:  # 信赖域太小
            return Trust_Region_Result(
                x, iter, delta, grad, success=False
            )  # pragma: no cover
        if iter > opts.max_iter:  # 迭代次数超过要求
            return Trust_Region_Result(
                x, iter, delta, grad, success=False
            )  # pragma: no cover

        if shaking <= 0 and not _hess_is_up_to_date:
            H = make_hess(x)

        # PCG
        step: Optional[ndarray]
        qpval: Optional[float]
        pcg_iter: int
        exit_flag: pcg.PCG_EXIT_FLAG
        step, qpval, pcg_iter, exit_flag = pcg.pcg(
            grad.value, H, _constr_shifted, delta
        )
        iter, shaking = iter + 1, shaking - 1

        if step is None:
            if _hess_is_up_to_date:
                delta /= 4.0
            else:
                H = make_hess(x)
            output(iter, fval, numpy.nan, grad.infnorm, pcg_iter, exit_flag, H)
            continue

        assert qpval is not None

        # 更新步长、试探点、试探函数值
        step_size: float = float(numpy.linalg.norm(step))  # type: ignore
        new_x: ndarray = x + step
        new_fval: float = objective(new_x)

        # 根据下降率确定信赖域缩放
        reduce: float = new_fval - fval
        ratio: float = 0 if reduce >= 0 else (1 if reduce <= qpval else reduce / qpval)
        if ratio >= 0.75 and step_size >= 0.9 * delta:
            delta *= 2
        elif ratio <= 0.25:
            if _hess_is_up_to_date:
                delta = step_size / 4.0
            else:
                H = make_hess(x)

        # 对符合下降要求的候选点进行更新
        if new_fval < fval:
            x, fval, _hess_is_up_to_date = new_x, new_fval, False
            grad, _constr_shifted = get_info(x, iter, grad.infnorm, init_grad_infnorm)
            if opts.abstol_fval is not None and old_fval - fval < opts.abstol_fval:
                stall_iter += 1
            else:
                old_fval, stall_iter = fval, 0

        output(iter, fval, step_size, grad.infnorm, pcg_iter, exit_flag, H)

        # 成功收敛准则
        if exit_flag == pcg.PCG_EXIT_FLAG.RESIDUAL_CONVERGENCE:  # PCG正定收敛
            if _hess_is_up_to_date:
                if grad.infnorm < opts.tol_grad:  # 梯度足够小
                    return Trust_Region_Result(x, iter, delta, grad, success=True)
                if step_size < opts.tol_step:  # 步长足够小
                    return Trust_Region_Result(x, iter, delta, grad, success=True)
            else:
                H = make_hess(x)

        if opts.max_stall_iter is not None and stall_iter >= opts.max_stall_iter:
            if _hess_is_up_to_date:
                return Trust_Region_Result(x, iter, delta, grad, success=True)
            else:
                H = make_hess(x)
