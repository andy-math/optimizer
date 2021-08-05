from typing import Final

from optimizer._internals.common.norm import norm_l2
from optimizer._internals.quad_prog.flag import Flag
from optimizer._internals.quad_prog.quad_eval import QuadEvaluator
from overloads.shortcuts import assertNoInfNaN
from overloads.typedefs import ndarray


class Status:
    x: Final[ndarray]
    fval: Final[float]
    angle: Final[float]
    flag: Final[Flag]
    size: Final[float]

    def __init__(
        self, x: ndarray, angle: float, flag: Flag, delta: float, qpval: QuadEvaluator
    ) -> None:
        assertNoInfNaN(x)
        self.x = x
        self.fval = qpval(x)
        self.angle = angle
        self.flag = flag
        self.size = norm_l2(x)
        assert self.size / delta < 1.0 + 1e-6
