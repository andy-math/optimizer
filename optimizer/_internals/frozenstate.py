# -*- coding: utf-8 -*-


from typing import Callable, NamedTuple, TextIO, Tuple

from optimizer._internals import options
from overloads.typing import ndarray


class FrozenState(NamedTuple):
    var_names: Tuple[str, ...]
    f: Callable[[ndarray], float]
    f_np: Callable[[ndarray], ndarray]
    g: Callable[[ndarray], ndarray]
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray]
    opts: options.Trust_Region_Options
    file: TextIO