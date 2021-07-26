import numpy
from optimizer._internals.common import findiff
from overloads import bind_checker, difference, dyn_typing
from overloads.shortcuts import assertNoInfNaN
from overloads.typedefs import ndarray

H = numpy.array([[2, 1], [1, 1]])
A = numpy.array([[1, 1]], dtype=numpy.float64)
b = numpy.array([1], dtype=numpy.float64)
lb = numpy.zeros((2,))
ub = numpy.full((2,), numpy.inf)


@dyn_typing.dyn_check_1(
    input=(dyn_typing.NDArray(numpy.float64, (dyn_typing.SizeConst(2),)),),
    output=dyn_typing.NDArray(numpy.float64, (dyn_typing.SizeConst(2),)),
)
@bind_checker.bind_checker_1(  # force line wrap
    input=bind_checker.make_checker_1(assertNoInfNaN),  # force line wrap
    output=assertNoInfNaN,  # force line wrap
)
def grad(x: ndarray) -> ndarray:
    x1, x2 = x
    assert x1 + x2 <= 1
    assert x1 >= 0
    assert x2 >= 0
    g = (H @ x) * 2.0
    return g  # type: ignore


hessian_GT: ndarray = 2 * H


def compare_hess(h: ndarray, gt: ndarray) -> None:
    assert difference.relative(h, gt) <= 1e-6
    assert difference.absolute(h, gt) <= 1e-6


class Test:
    def test_斜边中点(self) -> None:
        x = numpy.array([0.5, 0.5], dtype=numpy.float64)
        hessian = findiff.findiff(grad, x, (A, b, lb, ub))
        compare_hess(hessian, hessian_GT)

    def test_x轴顶点(self) -> None:
        x = numpy.array([1, 0], dtype=numpy.float64)
        hessian = findiff.findiff(grad, x, (A, b, lb, ub))
        compare_hess(hessian, hessian_GT * numpy.array([[1, 0], [1, 0]]))

    def test_y轴顶点(self) -> None:
        x = numpy.array([0, 1], dtype=numpy.float64)
        hessian = findiff.findiff(grad, x, (A, b, lb, ub))
        compare_hess(hessian, hessian_GT * numpy.array([[0, 1], [0, 1]]))

    def test_x轴中点(self) -> None:
        x = numpy.array([0.5, 0], dtype=numpy.float64)
        hessian = findiff.findiff(grad, x, (A, b, lb, ub))
        compare_hess(hessian, hessian_GT)

    def test_y轴中点(self) -> None:
        x = numpy.array([0, 0.5], dtype=numpy.float64)
        hessian = findiff.findiff(grad, x, (A, b, lb, ub))
        compare_hess(hessian, hessian_GT)

    def test_原点(self) -> None:
        x = numpy.array([0, 0], dtype=numpy.float64)
        hessian = findiff.findiff(grad, x, (A, b, lb, ub))
        compare_hess(hessian, hessian_GT)
