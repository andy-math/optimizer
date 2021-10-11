# -*- coding: utf-8 -*-


from typing import Callable, NamedTuple

from optimizer._internals.common import typing
from optimizer._internals.trust_region import options
from overloads.typedefs import ndarray


class FrozenState(NamedTuple):
    objective: typing.objective_t
    objective_np: Callable[[ndarray], ndarray]
    gradient: typing.gradient_t
    constraints: typing.constraints_t
    opts: options.Trust_Region_Options
