# -*- coding: utf-8 -*-

from typing import Callable, Final, NamedTuple, Tuple

import numpy
from numerical import linneq
from numerical.typedefs import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN

from optimizer import pcg
from optimizer._internals.trust_region import options
from optimizer._internals.trust_region.grad_maker import (
    Gradient,
    GradientCheck,
    make_gradient,
    make_hessian,
)

Trust_Region_Format_T = options.Trust_Region_Format_T
default_format = options.default_format
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
                dyn_typing.NDArray(numpy.float64, (N,)),
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
    class Hessian:
        shaking: Final[int] = (
            x.shape[0] if opts.shaking == "x.shape[0]" else opts.shaking
        )
        H: ndarray
        up_to_date: bool = True
        times: int = 0

        def __init__(self, x: ndarray) -> None:
            self.H = make_hessian(gradient, x, constraints, opts)

    def objective_ndarray(x: ndarray) -> ndarray:
        return numpy.array([objective(x)])

    def get_info(
        x: ndarray, check: GradientCheck
    ) -> Tuple[Gradient, Tuple[ndarray, ndarray, ndarray, ndarray]]:
        new_grad = make_gradient(gradient, x, constraints, opts, check=check)
        A, b, lb, ub = constraints
        _constr_shifted = (A, b - A @ x, lb - x, ub - x)
        return new_grad, _constr_shifted

    iter: int = 0
    delta: float = opts.init_delta
    assert linneq.check(x.reshape(-1, 1), constraints)

    fval: float = objective(x)
    hessian: Hessian = Hessian(x)
    grad: Gradient
    _constr_shifted: Tuple[ndarray, ndarray, ndarray, ndarray]

    grad, _constr_shifted = get_info(
        x, GradientCheck(objective_ndarray, iter, numpy.inf, 0.0)
    )

    options.output(iter, fval, grad.infnorm, None, hessian.H, opts, hessian.times)

    init_grad_infnorm: Final[float] = grad.infnorm
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

        if hessian.times > hessian.shaking and not hessian.up_to_date:
            hessian = Hessian(x)

        # PCG
        pcg_status = pcg.pcg(grad.value, hessian.H, _constr_shifted, delta)
        iter += 1
        hessian.times += 1

        if pcg_status.x is None:
            if hessian.up_to_date:
                delta /= 4.0
            else:
                hessian = Hessian(x)
            options.output(
                iter, fval, grad.infnorm, pcg_status, hessian.H, opts, hessian.times
            )
            continue

        assert pcg_status.fval is not None
        assert pcg_status.size is not None

        # 更新步长、试探点、试探函数值
        new_x: ndarray = x + pcg_status.x
        new_fval: float = objective(new_x)

        # 根据下降率确定信赖域缩放
        reduce: float = new_fval - fval
        ratio: float = (
            0
            if reduce >= 0
            else (1 if reduce <= pcg_status.fval else reduce / pcg_status.fval)
        )
        if ratio >= 0.75 and pcg_status.size >= 0.9 * delta:
            delta *= 2
        elif ratio <= 0.25:
            if not hessian.up_to_date:
                hessian = Hessian(x)
            else:
                delta = pcg_status.size / 4.0

        # 对符合下降要求的候选点进行更新
        if new_fval < fval:
            x, fval, hessian.up_to_date = new_x, new_fval, False
            grad, _constr_shifted = get_info(
                x,
                GradientCheck(objective_ndarray, iter, grad.infnorm, init_grad_infnorm),
            )
            # 下降量超过设定则重置延迟计数
            if opts.abstol_fval is not None and old_fval - fval < opts.abstol_fval:
                stall_iter += 1
            else:
                old_fval, stall_iter = fval, 0

        options.output(
            iter, fval, grad.infnorm, pcg_status, hessian.H, opts, hessian.times
        )

        # PCG正定收敛
        if pcg_status.flag == pcg.PCG_Flag.RESIDUAL_CONVERGENCE:
            if not hessian.up_to_date:
                hessian = Hessian(x)
                continue
            if grad.infnorm < opts.tol_grad:  # 梯度足够小
                return Trust_Region_Result(x, iter, delta, grad, success=True)
            if pcg_status.size < opts.tol_step:  # 步长足够小
                return Trust_Region_Result(x, iter, delta, grad, success=True)

        # 下降量过低收敛
        if opts.max_stall_iter is not None and stall_iter >= opts.max_stall_iter:
            if not hessian.up_to_date:
                hessian = Hessian(x)
                continue
            return Trust_Region_Result(x, iter, delta, grad, success=True)
