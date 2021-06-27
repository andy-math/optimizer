import enum


@enum.unique
class PCG_Flag(enum.Enum):
    RESIDUAL_CONVERGENCE = enum.auto()
    NEGATIVE_CURVATURE = enum.auto()
    OUT_OF_TRUST_REGION = enum.auto()
    VIOLATE_CONSTRAINTS = enum.auto()
