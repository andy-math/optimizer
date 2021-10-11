import enum
from typing import NamedTuple

from optimizer._internals.common.norm import norm_l2
from optimizer._internals.quad_prog.quad_eval import QuadEvaluator
from overloads.shortcuts import assertNoInfNaN
from overloads.typedefs import ndarray


@enum.unique
class Flag(enum.Enum):
    FATAL = enum.auto()
    INTERIOR = enum.auto()
    BOUNDARY = enum.auto()
    CONSTRAINT = enum.auto()


class Status(NamedTuple):
    x: ndarray
    fval: float
    angle: float
    flag: Flag
    size: float


def make_status(
    x: ndarray,
    angle: float,
    flag: Flag,
    delta: float,
    qpval: QuadEvaluator,
) -> Status:
    assertNoInfNaN(x)
    size = norm_l2(x)
    assert size / delta < 1.0 + 1e-6
    return Status(x=x, fval=qpval(x), angle=angle, flag=flag, size=size)
