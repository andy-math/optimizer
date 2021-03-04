# -*- coding: utf-8 -*-
from typing import Callable, Optional

import numpy
from mypy_extensions import NamedArg
from numerical import difference, findiff
from numerical.typedefs import ndarray

from optimizer import pcg


class Trust_Region_Options:
    tol_step: float = 1.0e-6
    tol_grad: float = 1.0e-6
    tol_func: float = 1.0e-6
    init_delta: float = 1.0
    max_iter: int
    format: Optional[
        Callable[
            [
                NamedArg(int, "iter"),
                NamedArg(float, "fval"),  # noqa: F821
                NamedArg(float, "step"),  # noqa: F821
                NamedArg(float, "grad"),  # noqa: F821
                NamedArg(float, "CGiter"),  # noqa: F821
            ],
            str,
        ]
    ] = "iter = {iter: 5d}, fval = {fval: 13.6g}, step = {step: 13.6g}, grad = {grad: 12.3g}, CG = {CGiter: 7d}".format  # noqa: E501

    def __init__(self, *, max_iter: int) -> None:
        self.max_iter = max_iter


class Trust_Region_Result:
    x: ndarray
    iter: int
    success: bool

    def __init__(self, x: ndarray, iter: int, *, success: bool) -> None:
        self.x = x
        self.iter = iter
        self.success = success


def trust_region(
    objective: Callable[[ndarray], float],
    gradient: Callable[[ndarray], ndarray],
    x: ndarray,
    opts: Trust_Region_Options,
) -> Trust_Region_Result:
    n = x.shape[0]
    constr_A = numpy.zeros((0, n))
    constr_b = numpy.zeros((0,))
    constr_lb = numpy.full((1, n), -numpy.inf)
    constr_ub = numpy.full((1, n), numpy.inf)

    assert opts.format is not None

    iter: int = 0
    pcg_iter: int = 0
    delta: float = opts.init_delta
    step_size: float = 0.0

    isposdef: Optional[bool] = None
    ratio: Optional[float] = None
    old_fval: Optional[float] = None

    fval: float = objective(x)
    grad: ndarray = gradient(x)
    H = findiff.findiff(gradient, x, constr_A, constr_b, constr_lb, constr_ub)

    while True:
        grad_infnorm: float = numpy.max(numpy.abs(grad))

        print(
            opts.format(
                iter=iter, fval=fval, step=step_size, grad=grad_infnorm, CGiter=pcg_iter
            )
        )

        if isposdef is not None and ratio is not None and old_fval is not None:
            reldiff = difference.relative(numpy.array([old_fval]), numpy.array([fval]))
            if grad_infnorm < opts.tol_grad and isposdef:
                return Trust_Region_Result(x, iter, success=True)
            if step_size < 0.9 * delta and ratio > 0.25 and reldiff < opts.tol_func:
                return Trust_Region_Result(x, iter, success=True)
            if step_size < opts.tol_step:
                return Trust_Region_Result(x, iter, success=True)
        if iter > opts.max_iter:
            return Trust_Region_Result(x, iter, success=False)

        old_fval = fval

        step: ndarray
        qpval: float
        wellDefined: bool
        step, qpval, isposdef, wellDefined, pcg_iter = pcg.pcg(grad, H, delta)

        new_x = x + step
        new_fval = objective(new_x)

        ratio = (new_fval - fval) / qpval
        if ratio >= 0.75 and step_size >= 0.9 * delta:
            delta *= 2
        elif ratio <= 0.25:
            delta = step_size / 4

        if wellDefined and new_fval < fval:
            x, fval, grad = new_x, new_fval, gradient(x)
            H = findiff.findiff(gradient, x, constr_A, constr_b, constr_lb, constr_ub)

        iter += 1
