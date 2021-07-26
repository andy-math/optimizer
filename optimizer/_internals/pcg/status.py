from optimizer._internals.common.norm import norm_l2
from optimizer._internals.pcg.flag import Flag
from optimizer._internals.pcg.qpval import qpval
from overloads.shortcuts import assertNoInfNaN
from overloads.typedefs import ndarray


class Status:
    x: ndarray
    fval: float
    iter: int
    flag: Flag
    size: float

    def __init__(
        self,
        x: ndarray,
        iter: int,
        flag: Flag,
        delta: float,
        g: ndarray,
        H: ndarray,
    ) -> None:
        assertNoInfNaN(x)
        self.x = x
        self.fval = qpval(g=g, H=H, x=x)
        self.iter = iter
        self.flag = flag
        self.size = norm_l2(x)
        assert self.size / delta < 1.0 + 1e-6
