# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, NamedTuple, Tuple

from numpy import ndarray
from optimizer._internals.trust_region import options


class FrozenState(NamedTuple):
    f: Callable[[ndarray], float]
    f_np: Callable[[ndarray], ndarray]
    g: Callable[[ndarray], ndarray]
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray]
    opts: options.Trust_Region_Options
