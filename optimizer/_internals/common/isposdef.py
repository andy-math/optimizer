# -*- coding: utf-8 -*-
import numpy
from numpy import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN

n = dyn_typing.SizeVar()


def noCheck(_: bool) -> None:
    pass


@dyn_typing.dyn_check_1(
    input=(dyn_typing.NDArray(numpy.float64, (n, n)),), output=dyn_typing.Bool()
)
@bind_checker.bind_checker_1(
    input=bind_checker.make_checker_1(assertNoInfNaN), output=noCheck
)
def isposdef(A: ndarray) -> bool:
    e: ndarray
    e, v = numpy.linalg.eig(A)  # type: ignore
    if e.dtype.type != numpy.float64:
        return False
    return bool(numpy.all(e > 0))
