import enum


@enum.unique
class Flag(enum.Enum):
    FATAL = enum.auto()
    INTERIOR = enum.auto()
    BOUNDARY = enum.auto()
    CONSTRAINT = enum.auto()
