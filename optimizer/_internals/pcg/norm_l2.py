import math

from overloads.typing import ndarray


def norm_l2(x: ndarray) -> float:
    return math.sqrt(float(x @ x))
