# -*- coding: utf-8 -*-
import numpy
from numerical.typedefs import ndarray
from optimizer import trust_region


def func(_x: ndarray) -> float:
    x: float = float(_x[0])
    y: float = float(_x[1])
    return 100 * (y - x * x) ** 2 + (1 - x) ** 2


def grad(_x: ndarray) -> ndarray:
    x: float = float(_x[0])
    y: float = float(_x[1])
    return numpy.array([-400 * (y - x * x) * x - 2 * (1 - x), 200 * (y - x * x)])


class Test_banana:
    def test1(self) -> None:
        n = 2
        constr_A = numpy.zeros((0, n))
        constr_b = numpy.zeros((0,))
        constr_lb = numpy.full((n,), -numpy.inf)
        constr_ub = numpy.full((n,), numpy.inf)

        opts = trust_region.Trust_Region_Options(max_iter=500)
        opts.shaking = -1
        result = trust_region.trust_region(
            func,
            grad,
            numpy.array([-1.9, 2]),
            (constr_A, constr_b, constr_lb, constr_ub),
            opts,
        )
        assert result.success
        assert numpy.all(numpy.round(result.x, 2) == 1)
        assert 27 < result.iter < 60


if __name__ == "__main__":
    Test_banana().test1()
