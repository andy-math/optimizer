# -*- coding: utf-8 -*-


from typing import Optional, Tuple

import numpy
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN, assertNoNaN
from overloads.typing import ndarray


def noCheck(_: bool) -> None:
    pass


def constraint_check(
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
    *,
    theta: Optional[ndarray] = None
) -> None:
    A, b, lb, ub = constraints
    if theta is not None:
        assertNoInfNaN(theta)
    assertNoInfNaN(A)
    assertNoInfNaN(b)
    assertNoNaN(lb)
    assertNoNaN(ub)


def _input_checker(
    parameters: Tuple[ndarray, Tuple[ndarray, ndarray, ndarray, ndarray]]
) -> None:
    theta, constraints = parameters
    constraint_check(constraints, theta=theta)


n = dyn_typing.SizeVar()
nConstraint = dyn_typing.SizeVar()


@dyn_typing.dyn_check_2(
    input=(
        dyn_typing.NDArray(numpy.float64, (n,)),
        dyn_typing.Tuple(
            (
                dyn_typing.NDArray(numpy.float64, (nConstraint, n)),
                dyn_typing.NDArray(numpy.float64, (nConstraint,)),
                dyn_typing.NDArray(numpy.float64, (n,)),
                dyn_typing.NDArray(numpy.float64, (n,)),
            )
        ),
    ),
    output=dyn_typing.Bool(),
)
@bind_checker.bind_checker_2(input=_input_checker, output=noCheck)
def check(
    theta: ndarray, constraints: Tuple[ndarray, ndarray, ndarray, ndarray]
) -> bool:
    A, b, lb, ub = constraints
    """检查参数theta是否满足约束[A @ theta <= b]，空约束返回True"""
    result = bool(
        numpy.all(lb <= theta) and numpy.all(theta <= ub) and numpy.all(A @ theta <= b)
    )
    return result


n = dyn_typing.SizeVar()
nConstraint = dyn_typing.SizeVar()


@dyn_typing.dyn_check_2(
    input=(
        dyn_typing.NDArray(numpy.float64, (n,)),
        dyn_typing.Tuple(
            (
                dyn_typing.NDArray(numpy.float64, (nConstraint, n)),
                dyn_typing.NDArray(numpy.float64, (nConstraint,)),
                dyn_typing.NDArray(numpy.float64, (n,)),
                dyn_typing.NDArray(numpy.float64, (n,)),
            )
        ),
    ),
    output=dyn_typing.Tuple(
        (
            dyn_typing.NDArray(numpy.float64, (n,)),
            dyn_typing.NDArray(numpy.float64, (n,)),
        )
    ),
)
@bind_checker.bind_checker_2(
    input=_input_checker, output=bind_checker.make_checker_2(assertNoNaN)
)
def margin(
    theta: ndarray, constraints: Tuple[ndarray, ndarray, ndarray, ndarray]
) -> Tuple[ndarray, ndarray]:
    """
    返回theta距离线性约束边界的间距下界和上界(h_lb, h_ub)
    h: 步长, lb: 下界, ub: 上界
    theta超出边界时 AssertionError
    """
    assert check(theta, constraints)
    A, b, lb, ub = constraints
    if b.shape[0] == 0:
        h_lb = numpy.full(theta.shape, -numpy.inf)
        h_ub = numpy.full(theta.shape, numpy.inf)
    else:
        """
        A @ (theta+h*(arange(n) == i)) == b
        => A @ h*(arange(n) == i) == b - A @ theta
        => h*A[:, i] == b - A @ theta (*must positive as valid point)
        => h == (b - A @ theta)/A[:, i]
        """
        residual: ndarray = b - A @ theta  # (nConst, )
        residual.shape = (A.shape[0], 1)  # (nConst, 1)
        h: ndarray = residual / A  # (nConst, n)
        """
        lb: 所有负数里面取最大
        ub: 所有正数里面取最小
        系数A为0，则约束与theta(i)无关
        """
        h_lb = h.copy()
        h_ub = h.copy()
        h_lb[A >= 0] = -numpy.inf
        h_ub[A <= 0] = numpy.inf
        h_lb = h_lb.max(axis=0)
        h_ub = h_ub.max(axis=0)
    """
    [lb/ub]补丁
    theta+h == [lb/ub]
    => h = [lb/ub]-theta
    """
    h_lb = numpy.maximum(h_lb, lb - theta)
    h_ub = numpy.minimum(h_ub, ub - theta)
    return h_lb, h_ub
