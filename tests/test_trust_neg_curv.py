# -*- coding: utf-8 -*-
import math

import numpy
from optimizer import trust_region
from overloads.typedefs import ndarray


def func(_x: ndarray) -> float:
    x: float = float(_x[0])
    return 1 / x + math.log(x)


def grad(_x: ndarray) -> ndarray:
    x: float = float(_x[0])
    return numpy.array([1 / x - 1 / (x * x)])


class Test_neg_curve:
    def test1(self) -> None:
        n = 1
        constr_A = numpy.zeros((0, n))
        constr_b = numpy.zeros((0,))
        constr_lb = numpy.array([0.25])
        constr_ub = numpy.array([10.0])

        opts = trust_region.Trust_Region_Options(max_iter=500)
        opts.check_rel = 1
        opts.check_abs = 1e-6
        result = trust_region.trust_region(
            func,
            grad,
            numpy.array([9.5]),
            (constr_A, constr_b, constr_lb, constr_ub),
            opts,
        )
        assert result.success
        assert numpy.all(result.x.round(6) == 1)
        assert 3 < result.iter < 20


if __name__ == "__main__":
    Test_neg_curve().test1()
