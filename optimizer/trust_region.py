# -*- coding: utf-8 -*-


from typing import Callable, Final, NamedTuple, Optional, Tuple

import numpy
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN
from overloads.typing import ndarray

from optimizer import pcg
from optimizer._internals.common import linneq
from optimizer._internals.common.gradient import Gradient
from optimizer._internals.common.hessian import Hessian
from optimizer._internals.trust_region import options
from optimizer._internals.trust_region.constr_preproc import constr_preproc
from optimizer._internals.trust_region.frozenstate import FrozenState
from optimizer._internals.trust_region.solution import Solution

Trust_Region_Format_T = options.Trust_Region_Format_T
default_format = options.default_format
Trust_Region_Options = options.Trust_Region_Options


class Trust_Region_Result(NamedTuple):
    x: ndarray
    iter: int
    delta: float
    gradient: Gradient
    hessian: Hessian
    success: bool


def _make_result(
    sol: Solution, iter: int, delta: float, *, success: bool
) -> Trust_Region_Result:
    return Trust_Region_Result(sol.x, iter, delta, sol.grad, sol.get_hessian(), success)


class _trust_region_impl:
    state: Final[FrozenState]

    def __init__(
        self,
        objective: Callable[[ndarray], float],
        gradient: Callable[[ndarray], ndarray],
        constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
        opts: Trust_Region_Options,
    ) -> None:
        def objective_ndarray(x: ndarray) -> ndarray:
            return numpy.array([objective(x)])

        constraints = constr_preproc(constraints)

        self.state = FrozenState(
            objective, objective_ndarray, gradient, constraints, opts
        )

    def _run(self, x: ndarray) -> Trust_Region_Result:
        assert linneq.check(x, self.state.constraints)

        hessian_force_shake: Optional[bool] = False

        sol0 = Solution(0, x, (numpy.inf, 0.0), self.state)

        iter, delta = 0, self.state.opts.init_delta
        old_fval, stall_iter = sol0.fval, 0

        hessian = sol0.get_hessian()

        options.output(
            iter,
            sol0.fval,
            sol0.grad.infnorm,
            None,
            hessian.ill,
            self.state.opts,
            1,
        )

        sol = sol0
        while True:
            # 失败情形的截止条件放在最前是因为pcg失败时的continue会导致后面代码被跳过
            if delta < self.state.opts.tol_step:  # 信赖域太小
                return _make_result(sol, iter, delta, success=False)
            if iter > self.state.opts.max_iter:  # 迭代次数超过要求
                return _make_result(sol, iter, delta, success=False)

            # hessian过期则重新采样
            assert hessian_force_shake is not None
            if hessian.times > hessian.max_times and not sol.hess_up_to_date:
                hessian = sol.get_hessian()
            elif hessian_force_shake:
                hessian = sol.get_hessian()
            hessian_force_shake = None

            # PCG
            pcg_status = pcg.pcg(sol.grad.value, hessian, sol.shifted_constr, delta)
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
                    hessian.ill,
                    self.state.opts,
                    hessian.times,
                )
                continue
            assert pcg_status.fval is not None
            assert pcg_status.size is not None

            # 更新步长、试探点、试探函数值
            new_sol = Solution(
                iter,
                sol.x + pcg_status.x,
                (sol.grad.infnorm, sol0.grad.infnorm),
                self.state,
            )

            # 根据下降率确定信赖域缩放
            reduce: float = new_sol.fval - sol.fval
            ratio = (
                0
                if reduce * pcg_status.fval <= 0
                else min(reduce / pcg_status.fval, 1.0)
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
            old_sol: Solution = sol
            if new_sol.fval < sol.fval:
                sol = new_sol
                # 下降量超过设定则重置延迟计数
                if (
                    self.state.opts.abstol_fval is not None
                    and old_fval - sol.fval < self.state.opts.abstol_fval
                ):
                    stall_iter += 1
                else:
                    old_fval, stall_iter = sol.fval, 0

            options.output(
                iter,
                sol.fval,
                sol.grad.infnorm,
                pcg_status,
                hessian.ill,
                self.state.opts,
                hessian.times,
            )

            # PCG正定收敛
            if pcg_status.flag in (pcg.Flag.RESIDUAL_CONVERGENCE, pcg.Flag.POLICY_ONLY):
                # 梯度足够小的case无关乎hessian信息
                if sol.grad.infnorm < self.state.opts.tol_grad:
                    return _make_result(sol, iter, delta, success=True)
                # 步长足够小的case要考虑hessian更新
                if pcg_status.size < self.state.opts.tol_step:
                    if not old_sol.hess_up_to_date:
                        assert hessian_force_shake or hessian_force_shake is None
                        hessian_force_shake = True
                        continue
                    return _make_result(sol, iter, delta, success=True)

            # 下降量过低的case要考虑hessian更新
            if (
                self.state.opts.max_stall_iter is not None
                and stall_iter >= self.state.opts.max_stall_iter
            ):
                if not old_sol.hess_up_to_date:
                    assert hessian_force_shake or hessian_force_shake is None
                    hessian_force_shake = True
                    continue
                return _make_result(sol, iter, delta, success=True)

            if hessian_force_shake is not None:
                assert hessian_force_shake
                continue
            hessian_force_shake = False


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
