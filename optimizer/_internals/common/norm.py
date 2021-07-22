import math
from typing import cast

import numpy
from overloads.typing import ndarray


def norm_inf(x: ndarray) -> float:
    return float(numpy.abs(x).max())


def norm_l2(x: ndarray) -> float:
    infnorm = norm_inf(x)
    if infnorm == 0:
        return 0
    x = cast(ndarray, x / infnorm)
    return infnorm * math.sqrt(float(x @ x))


def safe_normalize(x: ndarray) -> ndarray:
    l2norm = norm_l2(x)
    if l2norm == 0:
        return x
    return x / l2norm
