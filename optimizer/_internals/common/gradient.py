from typing import NamedTuple

from overloads.typing import ndarray


class Gradient(NamedTuple):
    value: ndarray
    infnorm: float
