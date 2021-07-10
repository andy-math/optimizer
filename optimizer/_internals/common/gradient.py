from typing import NamedTuple

from overloads.typing import ndarray


class RawGradient(NamedTuple):
    raw: ndarray


class Gradient(NamedTuple):
    value: ndarray
    infnorm: float
