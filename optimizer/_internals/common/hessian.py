import math
from typing import Optional

import numpy
from numpy import ndarray


class Hessian:
    value: ndarray
    ill: bool
    pinv: Optional[ndarray] = None

    def __init__(self, value: ndarray) -> None:
        _err = math.sqrt(float(numpy.finfo(numpy.float64).eps))

        value = (value.T + value) / 2.0

        e: ndarray = numpy.linalg.eig(value)[0]  # type: ignore
        if e.dtype != numpy.float64:
            e = numpy.real(e)
        assert e.dtype == numpy.float64

        min_e = float(numpy.min(e))

        self.value = value
        self.ill = min_e < _err

        if self.ill:
            self.pinv = numpy.linalg.pinv(value)  # type: ignore
