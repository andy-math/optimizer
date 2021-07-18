# -*- coding: utf-8 -*-
from typing import Callable, List, Tuple, cast

import numpy
from optimizer._internals import linneq
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN
from overloads.typing import ndarray


def findiff_check(
    parameters: Tuple[
        Callable[[ndarray], ndarray], ndarray, Tuple[ndarray, ndarray, ndarray, ndarray]
    ]
) -> None:
    _, theta, constraints = parameters
    linneq.constraint_check(constraints, theta=theta)


n = dyn_typing.SizeVar()
nFx = dyn_typing.SizeVar()
nConstraint = dyn_typing.SizeVar()


@dyn_typing.dyn_check_3(
    input=(
        dyn_typing.Callable(),
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
    output=dyn_typing.NDArray(numpy.float64, (nFx, n)),
)
@bind_checker.bind_checker_3(input=findiff_check, output=assertNoInfNaN)
def findiff(
    f: Callable[[ndarray], ndarray],
    theta: ndarray,
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray],
) -> ndarray:
    """
    基于有限差分返回f的雅可比矩阵[len(f) * len(x)]
    """
    """
    f(x0+a) = f(x0) + f'(x0)*a + 0.5*f''(x0)*a^2 + O(a^3)
    f(x0-b) = f(x0) - f'(x0)*b + 0.5*f''(x0)*b^2 - O(b^3)

    f(x0+a)-f(x0-b) = f'(x0)*(a+b) + 0.5*f''(x0)*(a^2-b^2) + O(a^3) + O(b^3)

    三阶精度条件：a^2-b^2 = 0     =>  a = b
                a^3 <= eps ≈ 0  =>  a <= eps^(1/3)
                b^3 <= eps ≈ 0  =>  b <= eps^(1/3)

                当 lb <= -b 且 a <= ub 时，a = b = eps^(1/3)

                当 -b <= lb 或 ub <= a 时，a = b = min(-lb, ub) >= 0，此时差分步长(a+b)太小，放宽到二阶精度

    二阶精度条件：a^2-b^2 <= eps ≈ 0
                a^3 <= eps ≈ 0      =>  a <= eps^(1/3)
                b^3 <= eps ≈ 0      =>  b <= eps^(1/3)

                不失一般性，设 a > b：a^2-b^2 <= eps  =>  a <= (eps+b^2)^(1/2)

                综上，a = min((eps+b^2)^(1/2), eps^(1/3))

    二阶精度当 b = 0 时，最小差分步长是 a-b = a = min(eps^(1/2), eps^(1/3)) = eps^(1/2)
    小于此步长，按无梯度处理
    """
    # 常量定义
    _eps = float(numpy.finfo(numpy.float64).eps)
    h_max = _eps ** (1 / 3)
    h_min = _eps ** (1 / 2)
    """
    差分使用的工具函数
    """
    fx_value = f(theta)
    assert len(fx_value.shape) == 1  # 函数返回值只能是向量，这样才满足ndims(Jacobian) == 2

    def _differ(i: int, h_forward: float, h_backward: float) -> ndarray:
        def _caller(h: float) -> ndarray:
            if h == 0:
                return fx_value
            theta_call = theta.copy()
            theta_call[i] += h
            return f(theta_call)

        if h_forward - h_backward < h_min:
            return numpy.zeros(fx_value.shape)
        result = (_caller(h_forward) - _caller(h_backward)) / (h_forward - h_backward)
        assert isinstance(result, numpy.ndarray)
        return result

    h_lb, h_ub = linneq.margin(theta, constraints)
    # patch 1 (lasso 兼容性): 对系数做差分的域不会越过0
    h_lb[theta > 0] = numpy.maximum(h_lb[theta > 0], -theta[theta > 0])
    h_ub[theta < 0] = numpy.minimum(h_ub[theta < 0], -theta[theta < 0])

    # patch 2 (内点兼容性)：对系数做差分的域总是内点上的
    h_lb = cast(ndarray, h_lb * (1.0 - 1.0e-4))
    h_ub = cast(ndarray, h_ub * (1.0 - 1.0e-4))

    Jacobian: List[ndarray] = []  # List(ndarray((m, )), n)
    for i in range(theta.shape[0]):
        h_forward = h_max
        h_backward = h_max
        if -h_backward <= h_lb[i] or h_ub[i] <= h_forward:
            if -h_lb[i] <= h_ub[i]:
                h_forward = min((_eps + h_lb[i] * h_lb[i]) ** (1 / 2), h_max)
            else:
                h_backward = min((_eps + h_ub[i] * h_ub[i]) ** (1 / 2), h_max)
        h_forward = min(h_forward, h_ub[i])
        h_backward = min(h_backward, -h_lb[i])
        Jacobian.append(_differ(i, h_forward, -h_backward))
    Jacobian_mat = numpy.stack(Jacobian, axis=1)
    return Jacobian_mat
