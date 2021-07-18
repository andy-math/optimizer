# -*- coding: utf-8 -*-
import numpy
from optimizer import trust_region
from overloads.typing import ndarray


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

        opts = trust_region.Trust_Region_Options(max_iter=99999, filename="banana.txt")
        result = trust_region.trust_region(
            func,
            grad,
            ["x1", "x2"],
            numpy.array([-1.9, 2]),
            (constr_A, constr_b, constr_lb, constr_ub),
            opts,
        )
        assert result.success
        assert numpy.all(result.x.round(5) == 1)
        assert 20 < result.iter < 20000


if __name__ == "__main__":
    Test_banana().test1()
