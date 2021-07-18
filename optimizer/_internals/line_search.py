import math
from typing import Final, List, Optional, Tuple

import numpy
from optimizer._internals.findiff import findiff
from optimizer._internals.frozenstate import FrozenState
from optimizer._internals.grad_maker import make_gradient
from optimizer._internals.solution import Solution
from overloads.typing import ndarray


class line_search:
    name: Final[str]
    old_fval: Final[float]
    x: Final[ndarray]
    i: Final[int]
    g: Final[float]
    lb: Final[float]
    ub: Final[float]
    state: Final[FrozenState]
    last_infnorm: Final[float]
    init_infnorm: Final[float]

    def __init__(self, i: int, sol: Solution, sol0: Solution) -> None:
        self.name = sol.state.var_names[i]
        self.old_fval = sol.fval
        self.x = sol.x
        self.i = i
        self.g = sol.gradient.value[i]
        self.lb = sol.lower_bound[i]
        self.ub = sol.upper_bound[i]
        self.state = sol.state
        self.last_infnorm = sol.gradient.infnorm
        self.init_infnorm = sol0.gradient.infnorm

    def _set(self, s: float) -> ndarray:
        x = self.x.copy()
        x[self.i] += s
        return x

    def run(
        self, iter: int, delta: float
    ) -> Tuple[float, Optional[Tuple[Solution, float]], str]:
        output: List[str] = [f"    iter = {iter}"]
        """
        解方程
        min f(x) ~ f(x) + g*s + 0.5*H*s^2
        其中g是f'(x), H是f''(x)
        """
        x = self.x[self.i]
        output.append(f"x = {x}")
        output.append(f"g = {self.g}")
        # 如果gradient小于tol_grad，pass此步
        if math.fabs(self.g) < self.state.opts.tol_grad:
            output.append("# abs(g) < tol_grad")
            output.append(f"[{x} -> None]")
            return delta, None, "\n    ".join(output)

        # 检查lb, ub，如果小于border_tol，pass此步
        output.append(f"lb = {self.lb}")
        output.append(f"ub = {self.ub}")
        output.append(f"border_abstol = {self.state.opts.border_abstol}")
        if self.g > 0 and -self.lb <= self.state.opts.border_abstol:
            output.append("# g > 0 and -lb <= border_abstol")
            output.append(f"[{x} -> None]")
            return delta, None, "\n    ".join(output)
        if self.g < 0 and self.ub <= self.state.opts.border_abstol:
            output.append("# g < 0 and ub <= border_abstol")
            output.append(f"[{x} -> None]")
            return delta, None, "\n    ".join(output)

        # findiff出H
        H = float(
            findiff(
                lambda s: make_gradient(
                    self.state.g,
                    self._set(float(s)),
                    self.state.constraints,
                    self.state.opts,
                    check=None,
                ).value,
                numpy.array([0.0]),
                (
                    numpy.empty((0, 1)),
                    numpy.empty((0,)),
                    numpy.array([self.lb]),
                    numpy.array([self.ub]),
                ),
            )[self.i]
        )
        output.append(f"H = {H}")

        # 对ax^2+bx+c在-g方向上求最大步长
        if H > 0:
            s = -self.g / H
        else:
            s = -math.inf if self.g > 0 else math.inf
        output.append(f"s = {s}")

        # 使用lb或ub限制最大步长
        if s > self.ub:
            output.append("# s > self.ub")
            s = 0.99 * self.ub
            output.append(f"s = {s}")
        if s < self.lb:
            output.append("# s < self.lb")
            s = 0.99 * self.lb
            output.append(f"s = {s}")

        # 使用delta限制最大步长
        output.append(f"delta = {delta}")
        assert delta > 0
        if math.fabs(s) > delta:
            output.append("# abs(s0) > delta")
            s = numpy.sign(s) * min(math.fabs(s), delta)
            output.append(f"s = {s}")
        qpval = self.g * s + (H * s * s) / 2
        output.append(f"qpval = {qpval}")
        assert qpval <= 0

        # 下降失败则收缩信赖域，提前返回
        output.append(f"old_fval = {self.old_fval}")
        new_fval = self.state.f(self._set(s))
        output.append(f"new_fval = {new_fval}")
        reduce = new_fval - self.old_fval
        output.append(f"reduce = {reduce}")
        if reduce > 0:
            output.append("# reduce >= 0")
            output.append("# delta /= 4")
            delta /= 4
            output.append(f"delta = {delta}")
            output.append(f"[{x} -> None]")
            return delta, None, "\n    ".join(output)

        # 代入f检验ratio
        if reduce <= qpval:
            output.append("# reduce <= qpval")
            ratio = 1.0
        else:
            ratio = reduce / qpval
        assert 0 <= ratio <= 1
        output.append(f"ratio = {ratio}")

        # 根据ratio结果调整delta
        if ratio >= 3 / 4 and math.fabs(s) >= 0.9 * delta:
            output.append("# ratio >= 3/4 and abs(s) >= 0.9*delta")
            output.append("# delta *= 2")
            delta *= 2
        elif ratio <= 1 / 4:
            output.append("# ratio <= 1/4")
            output.append("# delta = abs(s)/4")
            delta = math.fabs(s) / 4
        output.append(f"delta = {delta}")

        # 返回正确的x值
        output.append(f"[{x} -> {x+s}]")
        return (
            delta,
            (
                Solution(
                    iter,
                    new_fval,
                    self._set(s),
                    (self.last_infnorm, self.init_infnorm),
                    self.state,
                ),
                s,
            ),
            "\n    ".join(output),
        )
