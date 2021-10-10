import math
from typing import Final

import numpy

from overloads.typedefs import ndarray


class Hessian:
    value: Final[ndarray]
    ill: Final[bool]
    times: int = 0
    max_times: Final[int]

    def __init__(self, value: ndarray, *, max_times: int) -> None:
        _err = math.sqrt(float(numpy.finfo(numpy.float64).eps))

        value = (value.T + value) / 2.0
        e: ndarray = numpy.linalg.eigh(value)[0]
        assert e.dtype.type == numpy.float64
        min_e = float(e.min())

        self.value = value
        self.ill = min_e < _err
        self.max_times = max_times
