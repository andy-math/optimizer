# -*- coding: utf-8 -*-


import math
from typing import Final, NamedTuple, Optional, Tuple, Union

import numpy

from optimizer import quad_prog
from optimizer._internals.common import linneq, typing
from optimizer._internals.quad_prog.quad_eval import QuadEvaluator
from optimizer._internals.structures.frozenstate import FrozenState
from optimizer._internals.structures.gradient import Gradient
from optimizer._internals.structures.hessian import Hessian, make_hessian
from optimizer._internals.trust_region import format, options
from optimizer._internals.trust_region.constr_preproc import constr_preproc
from optimizer._internals.trust_region.solution import Solution, make_solution
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN
from overloads.typedefs import ndarray

Trust_Region_Options = options.Trust_Region_Options


class Trust_Region_Result(NamedTuple):
    x: ndarray
    iter: int
    delta: float
    gradient: Gradient
    success: bool


class _MutState(NamedTuple):
    iter: int
    delta: float
    old_fval: float
    stall_iter: int


def _make_result(
    sol: Solution,
    state: _MutState,
    *,
    success: bool,
) -> Trust_Region_Result:
    return Trust_Region_Result(sol.x, state.iter, state.delta, sol.grad, success)


def stop_criteria(
    state: FrozenState,
    mut_state: _MutState,
    old_sol: Solution,
    new_sol: Solution,
    pcg_status: quad_prog.Status,
    hessian_force_shake: bool,
) -> Union[bool, Trust_Region_Result]:
    if pcg_status.x is not None:
        assert pcg_status.fval is not None
        assert pcg_status.size is not None
        # PCG正定收敛
        if pcg_status.flag != quad_prog.Flag.FATAL:
            # 梯度足够小的case无关乎hessian信息
            if new_sol.grad.infnorm < state.opts.tol_grad:
                return _make_result(new_sol, mut_state, success=True)
            # 步长足够小的case要考虑hessian更新
            if pcg_status.size < state.opts.tol_step:
                if not old_sol.hess_up_to_date:
                    return True
                return _make_result(new_sol, mut_state, success=True)

    # 下降量过低的case要考虑hessian更新
    max_stall_iter: Final[Optional[int]] = state.opts.max_stall_iter
    if max_stall_iter is not None and mut_state.stall_iter >= max_stall_iter:
        if not old_sol.hess_up_to_date:
            return True
        return _make_result(new_sol, mut_state, success=True)
    # 信赖域太小
    if mut_state.delta < state.opts.tol_step:
        return _make_result(new_sol, mut_state, success=False)
    # 迭代次数超过要求
    if mut_state.iter >= state.opts.max_iter:
        return _make_result(new_sol, mut_state, success=False)
    return hessian_force_shake


def _output(
    sol: Solution,
    state: FrozenState,
    mut_state: _MutState,
    pcg_status: Optional[quad_prog.Status],
    hessian: Hessian,
) -> None:
    if state.opts.display:
        print(format.format(mut_state.iter, sol, hessian, pcg_status))


def _main_loop(
    sol0: Solution,
    old_sol: Solution,
    hessian: Hessian,
    state: FrozenState,
    mut_state: _MutState,
) -> Tuple[Solution, quad_prog.Status, bool, _MutState]:

    # PCG
    qp_eval = QuadEvaluator(g=old_sol.grad.value, H=hessian.value)
    pcg_status = quad_prog.quad_prog(qp_eval, old_sol.shifted_constr, mut_state.delta)
    hessian.times += 1

    # 更新步长、试探点、试探函数值
    new_sol = make_solution(
        mut_state.iter + 1,
        old_sol.x + pcg_status.x,
        (old_sol.grad.infnorm, sol0.grad.infnorm),
        state,
    )

    hessian_force_shake: bool

    # 根据下降率确定信赖域缩放
    reduce: float = new_sol.fval - old_sol.fval
    ratio = (
        0.0
        if reduce >= 0
        else (1.0 if reduce <= pcg_status.fval else reduce / pcg_status.fval)
    )
    delta = mut_state.delta
    if ratio >= 0.75 and pcg_status.size >= 0.9 * delta:
        delta *= 2
        hessian_force_shake = False
    elif ratio <= 0.25:
        if not old_sol.hess_up_to_date:
            hessian_force_shake = True
        else:
            delta = pcg_status.size / 4.0
            hessian_force_shake = False
    else:
        hessian_force_shake = False

    mut_state = _MutState(
        iter=mut_state.iter + 1,
        delta=delta,
        old_fval=mut_state.old_fval,
        stall_iter=mut_state.stall_iter,
    )

    return new_sol, pcg_status, hessian_force_shake, mut_state


def _run(state: FrozenState, x: ndarray) -> Trust_Region_Result:
    assert linneq.check(x, state.constraints)

    sol0 = make_solution(0, x, (numpy.inf, 0.0), state)
    assert sol0.fval != math.inf, "优化器迭代的起点函数值不能为inf"

    mut_state: _MutState = _MutState(
        iter=0,
        delta=state.opts.init_delta,
        old_fval=sol0.fval,
        stall_iter=0,
    )

    sol0, hessian = make_hessian(sol0, state)

    _output(sol0, state, mut_state, None, hessian)

    old_sol: Solution = sol0

    abstol_fval: Final[Optional[float]] = state.opts.abstol_fval
    while True:
        hessian_force_shake: bool
        new_sol, pcg_status, hessian_force_shake, mut_state = _main_loop(
            sol0, old_sol, hessian, state, mut_state
        )

        _output(new_sol, state, mut_state, pcg_status, hessian)

        result = stop_criteria(
            state, mut_state, old_sol, new_sol, pcg_status, hessian_force_shake
        )
        if isinstance(result, Trust_Region_Result):
            return result

        hessian_force_shake = result

        # 对符合下降要求的候选点进行更新
        if new_sol.fval < old_sol.fval:
            old_sol = new_sol
            # 下降量超过设定则重置延迟计数
            if abstol_fval is not None:
                if mut_state.old_fval - old_sol.fval < abstol_fval:
                    mut_state = _MutState(
                        iter=mut_state.iter,
                        delta=mut_state.delta,
                        old_fval=mut_state.old_fval,
                        stall_iter=mut_state.stall_iter + 1,
                    )
                else:
                    mut_state = _MutState(
                        iter=mut_state.iter,
                        delta=mut_state.delta,
                        old_fval=old_sol.fval,
                        stall_iter=0,
                    )

        # hessian过期则重新采样
        if hessian.times > hessian.max_times and not old_sol.hess_up_to_date:
            old_sol, hessian = make_hessian(old_sol, state)
        elif hessian_force_shake:
            old_sol, hessian = make_hessian(old_sol, state)


"""
过程式的接口
"""


def _input_check(
    input: Tuple[
        typing.objective_t,
        typing.gradient_t,
        ndarray,
        typing.constraints_t,
        Trust_Region_Options,
    ]
) -> None:
    _, _, x, constraints, _ = input
    linneq.constraint_check(constraints, theta=x)


def _output_check(output: Trust_Region_Result) -> None:
    assertNoInfNaN(output.x)


N = dyn_typing.SizeVar()


@dyn_typing.dyn_check_5(
    input=(
        dyn_typing.Callable(),
        dyn_typing.Callable(),
        dyn_typing.NDArray(numpy.float64, (N,)),
        typing.DynT_Constraints(N),
        dyn_typing.Class(Trust_Region_Options),
    ),
    output=dyn_typing.Class(Trust_Region_Result),
)
@bind_checker.bind_checker_5(input=_input_check, output=_output_check)
def trust_region(
    objective: typing.objective_t,
    gradient: typing.gradient_t,
    x: ndarray,
    constraints: typing.constraints_t,
    opts: Trust_Region_Options,
) -> Trust_Region_Result:
    format._format_times = 0

    state = FrozenState(
        objective=objective,
        objective_np=lambda x: numpy.array([objective(x)]),
        gradient=gradient,
        constraints=constr_preproc(constraints),
        opts=opts,
    )
    return _run(state, x)
