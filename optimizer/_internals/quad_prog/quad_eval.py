from typing import Final

import numpy

from overloads.shortcuts import assertNoInfNaN
from overloads.typedefs import ndarray


class QuadEvaluator:
    __H: Final[ndarray]
    __g: Final[ndarray]

    def __init__(self, *, g: ndarray, H: ndarray) -> None:
        (n,) = g.shape
        assert H.shape == (n, n)
        assert numpy.all(H.T == H)
        assertNoInfNaN(g)
        assertNoInfNaN(H)
        self.__g = g.copy()
        self.__H = H.copy()

    def __call__(self, x: ndarray) -> float:
        assertNoInfNaN(x)
        return 0.5 * float(x @ (self.__H @ x)) + float(self.__g @ x)

    @property
    def H(self) -> ndarray:
        return self.__H.copy()

    @property
    def g(self) -> ndarray:
        return self.__g.copy()
