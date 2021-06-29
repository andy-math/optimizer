# -*- coding: utf-8 -*-


from typing import Callable, Final, NamedTuple, Optional, Tuple

import numpy
from numpy import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN

from optimizer import pcg
from optimizer._internals.common import linneq
from optimizer._internals.common.hessian import Hessian
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
    class HProxy:
        shaking: Final[int] = (
            x.shape[0] if opts.shaking == "x.shape[0]" else opts.shaking
        )
        H: Hessian
        up_to_date: bool = True
        times: int = 0
        force_shake: bool = False

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

    assert linneq.check(x.reshape(-1, 1), constraints)

    fval = objective(x)
    hessian = HProxy(x)
    grad, _constr_shifted = get_info(
        x, GradientCheck(objective_ndarray, 0, numpy.inf, 0.0)
    )

    iter = 0
    delta = opts.init_delta
    init_grad_infnorm: Final[float] = grad.infnorm
    old_fval, stall_iter = fval, 0

    options.output(iter, fval, grad.infnorm, None, hessian.H.ill, opts, hessian.times)

    ratio: Optional[float] = None

    while True:
        # 失败情形的截止条件放在最前是因为pcg失败时的continue会导致后面代码被跳过
        if delta < opts.tol_step:  # 信赖域太小
            return Trust_Region_Result(
                x, iter, delta, grad, success=ratio is not None and ratio > 0.25
            )  # pragma: no cover
        if iter > opts.max_iter:  # 迭代次数超过要求
            return Trust_Region_Result(
                x, iter, delta, grad, success=False
            )  # pragma: no cover

        if (
            hessian.times > hessian.shaking and not hessian.up_to_date
        ) or hessian.force_shake:
            hessian = HProxy(x)

        # PCG
        pcg_status = pcg.pcg(grad.value, hessian.H, _constr_shifted, delta)
        iter += 1
        hessian.times += 1

        if pcg_status.x is None:
            ratio = None
            if hessian.up_to_date:
                delta /= 4.0
            else:
                hessian.force_shake = True
            options.output(
                iter, fval, grad.infnorm, pcg_status, hessian.H.ill, opts, hessian.times
            )
            continue

        assert pcg_status.fval is not None
        assert pcg_status.size is not None

        # 更新步长、试探点、试探函数值
        new_x: ndarray = x + pcg_status.x
        new_fval: float = objective(new_x)

        # 根据下降率确定信赖域缩放
        reduce: float = new_fval - fval
        ratio = (
            0 if reduce * pcg_status.fval <= 0 else min(reduce / pcg_status.fval, 1.0)
        )
        if ratio >= 0.75 and pcg_status.size >= 0.9 * delta and reduce < 0:
            delta *= 2
        elif ratio <= 0.25 or reduce > 0:
            if not hessian.up_to_date:
                hessian.force_shake = True
            else:
                delta = pcg_status.size / 4.0

        success: bool = False
        # 对符合下降要求的候选点进行更新
        if new_fval < fval:
            x, fval, success = new_x, new_fval, True
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
            iter, fval, grad.infnorm, pcg_status, hessian.H.ill, opts, hessian.times
        )

        # PCG正定收敛
        if pcg_status.flag in (pcg.Flag.RESIDUAL_CONVERGENCE, pcg.Flag.POLICY_ONLY):
            # 梯度足够小的case无关乎hessian信息
            if grad.infnorm < opts.tol_grad:
                return Trust_Region_Result(x, iter, delta, grad, success=True)
            # 步长足够小的case要考虑hessian更新
            if pcg_status.size < opts.tol_step:
                if not hessian.up_to_date:
                    hessian.force_shake = True
                    continue
                return Trust_Region_Result(x, iter, delta, grad, success=True)

        # 下降量过低的case要考虑hessian更新
        if opts.max_stall_iter is not None and stall_iter >= opts.max_stall_iter:
            if not hessian.up_to_date:
                hessian.force_shake = True
                continue
            return Trust_Region_Result(x, iter, delta, grad, success=True)

        if success:
            hessian.up_to_date = False
