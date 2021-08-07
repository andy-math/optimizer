# -*- coding: utf-8 -*-


import math
from typing import Callable, Final, NamedTuple, Optional, Tuple, Union

import numpy

from optimizer import quad_prog
from optimizer._internals.common import linneq
from optimizer._internals.common.gradient import Gradient
from optimizer._internals.common.hessian import Hessian
from optimizer._internals.quad_prog.quad_eval import QuadEvaluator
from optimizer._internals.trust_region import format, options
from optimizer._internals.trust_region.constr_preproc import constr_preproc
from optimizer._internals.trust_region.frozenstate import FrozenState
from optimizer._internals.trust_region.solution import Solution
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN
from overloads.typedefs import ndarray

Trust_Region_Options = options.Trust_Region_Options


class Trust_Region_Result(NamedTuple):
    x: ndarray
    iter: int
    delta: float
    gradient: Gradient
    hessian: Hessian
    success: bool


class _trust_region_impl:
    abstol_fval: Final[Optional[float]]
    max_stall_iter: Final[Optional[int]]
    state: Final[FrozenState]
    iter: int
    delta: float
    old_fval: float
    stall_iter: int

    def __init__(
        self,
        objective: Callable[[ndarray], float],
        gradient: Callable[[ndarray], ndarray],
        constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
        opts: Trust_Region_Options,
    ) -> None:
        format._format_times = 0

        def objective_ndarray(x: ndarray) -> ndarray:
            return numpy.array([objective(x)])

        constraints = constr_preproc(constraints)

        self.abstol_fval = opts.abstol_fval
        self.max_stall_iter = opts.max_stall_iter
        self.state = FrozenState(
            objective, objective_ndarray, gradient, constraints, opts
        )

    def _make_result(self, sol: Solution, *, success: bool) -> Trust_Region_Result:
        return Trust_Region_Result(
            sol.x, self.iter, self.delta, sol.grad, sol.get_hessian(), success
        )

    def _output(
        self, sol: Solution, pcg_status: Optional[quad_prog.Status], hessian: Hessian
    ) -> None:
        if self.state.opts.display:
            print(format.format(self.iter, sol, hessian, pcg_status))

    def stop_criteria(
        self,
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
                if new_sol.grad.infnorm < self.state.opts.tol_grad:
                    return self._make_result(new_sol, success=True)
                # 步长足够小的case要考虑hessian更新
                if pcg_status.size < self.state.opts.tol_step:
                    if not old_sol.hess_up_to_date:
                        return True
                    return self._make_result(new_sol, success=True)

        # 下降量过低的case要考虑hessian更新
        if self.max_stall_iter is not None and self.stall_iter >= self.max_stall_iter:
            if not old_sol.hess_up_to_date:
                return True
            return self._make_result(new_sol, success=True)
        # 信赖域太小
        if self.delta < self.state.opts.tol_step:
            return self._make_result(new_sol, success=False)
        # 迭代次数超过要求
        if self.iter >= self.state.opts.max_iter:
            return self._make_result(new_sol, success=False)
        return hessian_force_shake

    def _main_loop(
        self, sol0: Solution, old_sol: Solution, hessian: Hessian
    ) -> Tuple[Solution, quad_prog.Status, bool]:

        # PCG
        qp_eval = QuadEvaluator(g=old_sol.grad.value, H=hessian.value)
        pcg_status = quad_prog.quad_prog(qp_eval, old_sol.shifted_constr, self.delta)
        self.iter += 1
        hessian.times += 1

        # 更新步长、试探点、试探函数值
        new_sol = Solution(
            self.iter,
            old_sol.x + pcg_status.x,
            (old_sol.grad.infnorm, sol0.grad.infnorm),
            self.state,
        )

        hessian_force_shake: bool

        # 根据下降率确定信赖域缩放
        reduce: float = new_sol.fval - old_sol.fval
        ratio = (
            0.0
            if reduce >= 0
            else (1.0 if reduce <= pcg_status.fval else reduce / pcg_status.fval)
        )
        if ratio >= 0.75 and pcg_status.size >= 0.9 * self.delta:
            self.delta *= 2
            hessian_force_shake = False
        elif ratio <= 0.25:
            if not old_sol.hess_up_to_date:
                hessian_force_shake = True
            else:
                self.delta = pcg_status.size / 4.0
                hessian_force_shake = False
        else:
            hessian_force_shake = False

        return new_sol, pcg_status, hessian_force_shake

    def _run(self, x: ndarray) -> Trust_Region_Result:
        assert linneq.check(x, self.state.constraints)

        sol0 = Solution(0, x, (numpy.inf, 0.0), self.state)
        assert sol0.fval != math.inf, "优化器迭代的起点函数值不能为inf"

        self.iter = 0
        self.delta = self.state.opts.init_delta
        self.old_fval = sol0.fval
        self.stall_iter = 0

        hessian = sol0.get_hessian()

        self._output(sol0, None, hessian)

        old_sol: Solution = sol0

        while True:
            hessian_force_shake: bool
            new_sol, pcg_status, hessian_force_shake = self._main_loop(
                sol0, old_sol, hessian
            )

            self._output(new_sol, pcg_status, hessian)

            result = self.stop_criteria(
                old_sol, new_sol, pcg_status, hessian_force_shake
            )
            if isinstance(result, Trust_Region_Result):
                return result

            hessian_force_shake = result

            # 对符合下降要求的候选点进行更新
            if new_sol.fval < old_sol.fval:
                old_sol = new_sol
                # 下降量超过设定则重置延迟计数
                if self.abstol_fval is not None:
                    if self.old_fval - old_sol.fval < self.abstol_fval:
                        self.stall_iter += 1
                    else:
                        self.old_fval, self.stall_iter = old_sol.fval, 0

            # hessian过期则重新采样
            if hessian.times > hessian.max_times and not old_sol.hess_up_to_date:
                hessian = old_sol.get_hessian()
            elif hessian_force_shake:
                hessian = old_sol.get_hessian()


"""
过程式的接口
"""


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
    return _trust_region_impl(objective, gradient, constraints, opts)._run(x)
