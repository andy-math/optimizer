from typing import Callable, NewType, Tuple

import numpy

from overloads import dyn_typing
from overloads.typedefs import ndarray

objective_t = Callable[[ndarray], float]
gradient_t = Callable[[ndarray], ndarray]
constraints_t = Tuple[ndarray, ndarray, ndarray, ndarray]
proj_t = NewType("proj_t", ndarray)


def DynT_Constraints(N: dyn_typing.DepSize) -> dyn_typing.DepType:
    nConstraint = dyn_typing.SizeVar()
    return dyn_typing.Tuple(
        (
            dyn_typing.NDArray(numpy.float64, (nConstraint, N)),
            dyn_typing.NDArray(numpy.float64, (nConstraint,)),
            dyn_typing.NDArray(numpy.float64, (N,)),
            dyn_typing.NDArray(numpy.float64, (N,)),
        )
    )
