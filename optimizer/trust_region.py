# -*- coding: utf-8 -*-

from __future__ import annotations

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


class _FrozenState(NamedTuple):
    f: Callable[[ndarray], float]
    f_np: Callable[[ndarray], ndarray]
    g: Callable[[ndarray], ndarray]
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray]
    opts: Trust_Region_Options


class _HessianProxy:
    value: Hessian
    times: int = 0
    max_times: int

    def __init__(self, value: Hessian, max_times: int) -> None:
        self.value = value
        self.max_times = max_times


class _Solution:
    state: Final[_FrozenState]
    fval: Final[float]
    x: Final[ndarray]
    grad: Final[Gradient]
    shifted_constr: Final[Tuple[ndarray, ndarray, ndarray, ndarray]]
    hess_up_to_date: bool = False

    def __init__(
        self,
        iter: int,
        x: ndarray,
        g_infnorm: Tuple[(float, float)],
        state: _FrozenState,
    ) -> None:
        self.state = state
        self.fval = state.f(x)
        self.x = x
        grad = make_gradient(
            state.g,
            x,
            state.constraints,
            state.opts,
            check=GradientCheck(state.f_np, iter, *g_infnorm),
        )
        self.grad = grad
        A, b, lb, ub = state.constraints
        self.shifted_constr = (A, b - A @ x, lb - x, ub - x)

    def get_hessian(self) -> _HessianProxy:
        self.hess_up_to_date = True
        return _HessianProxy(
            make_hessian(self.state.g, self.x, self.state.constraints, self.state.opts),
            self.x.shape[0]
            if self.state.opts.shaking == "x.shape[0]"
            else self.state.opts.shaking,
        )


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
    _objective: Callable[[ndarray], float],
    _gradient: Callable[[ndarray], ndarray],
    _x: ndarray,
    _constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    _opts: Trust_Region_Options,
) -> Trust_Region_Result:
    def objective_ndarray(x: ndarray) -> ndarray:
        return numpy.array([_objective(x)])

    assert linneq.check(_x.reshape(-1, 1), _constraints)

    hessian_force_shake: Optional[bool] = False

    state = _FrozenState(_objective, objective_ndarray, _gradient, _constraints, _opts)
    sol0 = _Solution(0, _x, (numpy.inf, 0.0), state)

    iter, delta = 0, state.opts.init_delta
    old_fval, stall_iter = sol0.fval, 0

    hessian = sol0.get_hessian()

    options.output(
        iter, sol0.fval, sol0.grad.infnorm, None, hessian.value.ill, state.opts, 1
    )

    sol = sol0
    while True:
        # 失败情形的截止条件放在最前是因为pcg失败时的continue会导致后面代码被跳过
        if delta < state.opts.tol_step:  # 信赖域太小
            return Trust_Region_Result(sol.x, iter, delta, sol.grad, success=False)
        if iter > state.opts.max_iter:  # 迭代次数超过要求
            return Trust_Region_Result(sol.x, iter, delta, sol.grad, success=False)

        # hessian过期则重新采样
        assert hessian_force_shake is not None
        if hessian.times > hessian.max_times and not sol.hess_up_to_date:
            hessian = sol.get_hessian()
        elif hessian_force_shake:
            hessian = sol.get_hessian()
        hessian_force_shake = None

        # PCG
        pcg_status = pcg.pcg(sol.grad.value, hessian.value, sol.shifted_constr, delta)
        iter += 1
        hessian.times += 1

        # PCG失败recover
        if pcg_status.x is None:
            assert hessian_force_shake is None
            if not sol.hess_up_to_date:
                hessian_force_shake = True
            else:
                hessian_force_shake = False
                delta /= 4.0
            options.output(
                iter,
                sol.fval,
                sol.grad.infnorm,
                pcg_status,
                hessian.value.ill,
                state.opts,
                hessian.times,
            )
            continue
        assert pcg_status.fval is not None
        assert pcg_status.size is not None

        # 更新步长、试探点、试探函数值
        new_sol = _Solution(
            iter, sol.x + pcg_status.x, (sol.grad.infnorm, sol0.grad.infnorm), state
        )

        # 根据下降率确定信赖域缩放
        reduce: float = new_sol.fval - sol.fval
        ratio = (
            0 if reduce * pcg_status.fval <= 0 else min(reduce / pcg_status.fval, 1.0)
        )
        if ratio >= 0.75 and pcg_status.size >= 0.9 * delta and reduce < 0:
            delta *= 2
        elif ratio <= 0.25 or reduce > 0:
            if not sol.hess_up_to_date:
                assert hessian_force_shake is None
                hessian_force_shake = True
            else:
                delta = pcg_status.size / 4.0

        # 对符合下降要求的候选点进行更新
        old_sol: _Solution = sol
        if new_sol.fval < sol.fval:
            sol = new_sol
            # 下降量超过设定则重置延迟计数
            if (
                state.opts.abstol_fval is not None
                and old_fval - sol.fval < state.opts.abstol_fval
            ):
                stall_iter += 1
            else:
                old_fval, stall_iter = sol.fval, 0

        options.output(
            iter,
            sol.fval,
            sol.grad.infnorm,
            pcg_status,
            hessian.value.ill,
            state.opts,
            hessian.times,
        )

        # PCG正定收敛
        if pcg_status.flag in (pcg.Flag.RESIDUAL_CONVERGENCE, pcg.Flag.POLICY_ONLY):
            # 梯度足够小的case无关乎hessian信息
            if sol.grad.infnorm < state.opts.tol_grad:
                return Trust_Region_Result(sol.x, iter, delta, sol.grad, success=True)
            # 步长足够小的case要考虑hessian更新
            if pcg_status.size < state.opts.tol_step:
                if not old_sol.hess_up_to_date:
                    assert hessian_force_shake or hessian_force_shake is None
                    hessian_force_shake = True
                    continue
                return Trust_Region_Result(sol.x, iter, delta, sol.grad, success=True)

        # 下降量过低的case要考虑hessian更新
        if (
            state.opts.max_stall_iter is not None
            and stall_iter >= state.opts.max_stall_iter
        ):
            if not old_sol.hess_up_to_date:
                assert hessian_force_shake or hessian_force_shake is None
                hessian_force_shake = True
                continue
            return Trust_Region_Result(sol.x, iter, delta, sol.grad, success=True)

        if hessian_force_shake is not None:
            assert hessian_force_shake
            continue
        hessian_force_shake = False
