# -*- coding: utf-8 -*-


from typing import Callable, NamedTuple, Tuple

from optimizer._internals.trust_region import options
from overloads.typedefs import ndarray


class FrozenState(NamedTuple):
    f: Callable[[ndarray], float]
    f_np: Callable[[ndarray], ndarray]
    g: Callable[[ndarray], ndarray]
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray]
    opts: options.Trust_Region_Options
