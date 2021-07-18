# -*- coding: utf-8 -*-
import math

import numpy
from optimizer import trust_region
from optimizer._internals.trust_region.grad_check import Grad_Check_Failed
from overloads import difference
from overloads.capture_exceptions import Captured_Exception, capture_exceptions
from overloads.typing import ndarray


def func(_x: ndarray) -> float:
    x: float = float(_x[0])
    return 1 / x + math.log(x)


def grad(_x: ndarray) -> ndarray:
    x: float = float(_x[0])
    return numpy.array([1 / x + 1 / (x * x)])


n = 1
constr_A = numpy.zeros((0, n))
constr_b = numpy.zeros((0,))
constr_lb = numpy.array([0.25])
constr_ub = numpy.array([10.0])


def run(opts: trust_region.Trust_Region_Options) -> trust_region.Trust_Region_Result:
    return trust_region.trust_region(
        func,
        grad,
        numpy.array([9.5]),
        (constr_A, constr_b, constr_lb, constr_ub),
        opts,
    )


class Test_grad_check:
    def test1(self) -> None:
        opts = trust_region.Trust_Region_Options(max_iter=500)
        result = capture_exceptions(run, opts, catch=Grad_Check_Failed)
        assert isinstance(result, Captured_Exception)
        assert isinstance(result.exception, Grad_Check_Failed)
        assert result.exception.checker == difference.relative

    def test2(self) -> None:
        opts = trust_region.Trust_Region_Options(max_iter=500)
        opts.check_rel = numpy.inf
        opts.check_abs = 1.0e-6
        result = capture_exceptions(run, opts, catch=Grad_Check_Failed)
        assert isinstance(result, Captured_Exception)
        assert isinstance(result.exception, Grad_Check_Failed)
        assert result.exception.checker == difference.absolute


if __name__ == "__main__":
    Test_grad_check().test1()
    Test_grad_check().test2()
