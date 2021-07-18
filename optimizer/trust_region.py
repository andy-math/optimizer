# -*- coding: utf-8 -*-


import math
import os
from typing import Callable, Final, List, NamedTuple, Optional, Tuple

import numpy
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN
from overloads.typing import ndarray

from optimizer._internals import linneq, options
from optimizer._internals.constr_preproc import constr_preproc
from optimizer._internals.frozenstate import FrozenState
from optimizer._internals.grad_maker import Gradient
from optimizer._internals.line_search import line_search
from optimizer._internals.solution import Solution

Trust_Region_Options = options.Trust_Region_Options


class Trust_Region_Result(NamedTuple):
    x: ndarray
    iter: int
    delta: List[float]
    gradient: Gradient
    hessian: ndarray
    success: bool


class _trust_region_impl:
    abstol_fval: Final[Optional[float]]
    max_stall_iter: Final[Optional[int]]
    state: Final[FrozenState]
    iter: int
    delta: List[float]
    old_fval: float
    stall_iter: int

    def __init__(
        self,
        var_names: Tuple[str, ...],
        objective: Callable[[ndarray], float],
        gradient: Callable[[ndarray], ndarray],
        constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
        opts: Trust_Region_Options,
    ) -> None:
        def objective_ndarray(x: ndarray) -> ndarray:
            return numpy.array([objective(x)])

        constraints = constr_preproc(constraints)

        self.abstol_fval = opts.abstol_fval
        self.max_stall_iter = opts.max_stall_iter

        path, _ = os.path.split(opts.filename)
        if path != "":
            os.makedirs(path, exist_ok=True)
        file = open(opts.filename, "tw", encoding="utf-8")

        self.state = FrozenState(
            var_names, objective, objective_ndarray, gradient, constraints, opts, file
        )

    def _make_result(self, sol: Solution, *, success: bool) -> Trust_Region_Result:
        return Trust_Region_Result(
            sol.x, self.iter, self.delta, sol.gradient, sol.get_hessian(), success
        )

    def _print(self, s: str) -> None:
        print(s, file=self.state.file)
        if self.state.opts.display:
            print(s)

    def stop_criteria(
        self, sol: Solution, max_step: float, changed: bool
    ) -> Optional[Trust_Region_Result]:
        if changed:
            # 梯度足够小
            if sol.gradient.infnorm < self.state.opts.tol_grad:
                self._print("梯度足够小")
                return self._make_result(sol, success=True)
            # 步长足够小
            if max_step < self.state.opts.tol_step:
                self._print("步长足够小")
                return self._make_result(sol, success=True)
            # 下降量过低
            if (
                self.max_stall_iter is not None
                and self.stall_iter >= self.max_stall_iter
            ):
                self._print("下降量过低")
                return self._make_result(sol, success=True)
        # 信赖域太小
        if max(self.delta) < self.state.opts.tol_step:
            self._print("信赖域太小")
            return self._make_result(sol, success=False)
        # 迭代次数超过要求
        if self.iter > self.state.opts.max_iter:
            self._print("迭代次数超过要求")
            return self._make_result(sol, success=False)
        return None

    def _main_loop(self, sol0: Solution, sol: Solution) -> Tuple[Solution, float, bool]:
        max_step = 0.0
        changed = False
        for i in range(len(self.state.var_names)):
            search = line_search(i, sol, sol0)
            self.iter += 1
            self.delta[i], new_sol, info = search.run(self.iter, self.delta[i])
            self._print(self.state.var_names[i])
            self._print(info)
            if new_sol is not None:
                changed = True
                max_step = max(max_step, math.fabs(new_sol[1]))
                sol = new_sol[0]
                # 下降量超过设定则重置延迟计数
                if (
                    self.abstol_fval is not None
                    and self.old_fval - sol.fval < self.abstol_fval
                ):
                    self.stall_iter += 1
                else:
                    self.old_fval, self.stall_iter = sol.fval, 0
        return sol, max_step, changed

    def _run(self, x: ndarray) -> Trust_Region_Result:
        assert linneq.check(x, self.state.constraints)

        sol0 = Solution(0, self.state.f(x), x, (numpy.inf, 0.0), self.state)

        self.iter = 0
        self.delta = [
            self.state.opts.init_delta for _ in range(len(self.state.var_names))
        ]
        self.old_fval = sol0.fval
        self.stall_iter = 0

        sol = sol0

        while True:
            sol, max_step, changed = self._main_loop(sol0, sol)
            result = self.stop_criteria(sol, max_step, changed)
            if result is not None:
                return result

            self._print(str(sol.x))
            self._print(str(sol.gradient.value))


"""
过程式的接口
"""


def _input_check(
    input: Tuple[
        Callable[[ndarray], float],
        Callable[[ndarray], ndarray],
        List[str],
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
    _, _, _, x, constraints, _ = input
    linneq.constraint_check(constraints, theta=x)


def _output_check(output: Trust_Region_Result) -> None:
    assertNoInfNaN(output.x)


N = dyn_typing.SizeVar()
nConstraint = dyn_typing.SizeVar()


@dyn_typing.dyn_check_6(
    input=(
        dyn_typing.Callable(),
        dyn_typing.Callable(),
        dyn_typing.List(dyn_typing.Str(), N),
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
@bind_checker.bind_checker_6(input=_input_check, output=_output_check)
def trust_region(
    objective: Callable[[ndarray], float],
    gradient: Callable[[ndarray], ndarray],
    var_names: List[str],
    x: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    opts: Trust_Region_Options,
) -> Trust_Region_Result:
    return _trust_region_impl(
        tuple(var_names), objective, gradient, constraints, opts
    )._run(x)
