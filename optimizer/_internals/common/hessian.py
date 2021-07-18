import math
from typing import Final, Optional

import numpy
from overloads.typing import ndarray


class Hessian:
    value: Final[ndarray]
    ill: Final[bool]
    pinv: Final[Optional[ndarray]] = None
    times: int = 0
    max_times: Final[int]

    def __init__(self, value: ndarray, *, max_times: int) -> None:
        _err = math.sqrt(float(numpy.finfo(numpy.float64).eps))

        value = (value.T + value) / 2.0  # type: ignore

        e: ndarray = numpy.linalg.eig(value)[0]  # type: ignore
        if e.dtype.type != numpy.float64:
            e = e.real
        assert e.dtype.type == numpy.float64

        min_e = float(e.min())

        self.value = value
        self.ill = min_e < _err

        if self.ill:
            self.pinv = numpy.linalg.pinv(value)  # type: ignore

        self.max_times = max_times
