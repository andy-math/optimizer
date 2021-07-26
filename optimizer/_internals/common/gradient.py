from typing import NamedTuple

from overloads.typedefs import ndarray


class Gradient(NamedTuple):
    value: ndarray
    infnorm: float
