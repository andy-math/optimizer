# -*- coding: utf-8 -*-
import numpy
from optimizer import trust_region
from overloads.typing import ndarray

"""
f = x^2 + (y-2)^2 = x^2 + y^2 - 4y + 4
-x-1 <= y <=  x+1
 x-1 <= y <= -x+1

-x-y <= 1
-x+y <= 1
 x-y <= 1
 y-x <= 1
"""


class Test_constraints:
    def test1(self) -> None:
        def func(_x: ndarray) -> float:
            x: float
            y: float
            x, y = _x
            return x * x + (y - 2) * (y - 2)

        def grad(_x: ndarray) -> ndarray:
            x: float
            y: float
            x, y = _x
            return numpy.array([2 * x, 2 * y - 4])

        constr_A = numpy.array(
            [
                [-1, -1],
                [1, -1],
                [-1, 1],
                [1, 1],
            ],
            dtype=numpy.float64,
        )
        constr_b = numpy.ones((4,))
        constr_lb = numpy.full((2,), -numpy.inf)
        constr_ub = numpy.full((2,), numpy.inf)

        opts = trust_region.Trust_Region_Options(max_iter=500)
        opts.check_rel = 1
        opts.check_abs = 1e-6
        result = trust_region.trust_region(
            func,
            grad,
            numpy.array([0.9, 0.0]),
            (constr_A, constr_b, constr_lb, constr_ub),
            opts,
        )
        # assert result.success
        assert numpy.all(result.x.round(6) == numpy.array([0.0, 1.0]))
        assert 3 < result.iter < 40

    def test2(self) -> None:
        def func(_x: ndarray) -> float:
            x: float
            y: float
            x, y = _x
            return (x + 2) * (x + 2) + (y - 2) * (y - 2)

        def grad(_x: ndarray) -> ndarray:
            x: float
            y: float
            x, y = _x
            return numpy.array([2 * (x + 2), 2 * (y - 2)])

        constr_A = numpy.empty((0, 2))
        constr_b = numpy.ones((0,))
        constr_lb = numpy.array([-1, -1], dtype=numpy.float64)
        constr_ub = numpy.array([1, 1], dtype=numpy.float64)

        opts = trust_region.Trust_Region_Options(max_iter=500)
        opts.check_rel = 1
        opts.check_abs = 1e-6
        result = trust_region.trust_region(
            func,
            grad,
            numpy.array([0.9, 0.9]),
            (constr_A, constr_b, constr_lb, constr_ub),
            opts,
        )
        assert result.success
        assert numpy.all(result.x.round(6) == numpy.array([-1.0, 1.0]))
        assert 3 < result.iter < 40


if __name__ == "__main__":
    Test_constraints().test1()
    Test_constraints().test2()
