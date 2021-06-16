# -*- coding: utf-8 -*-
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy
from mypy_extensions import NamedArg
from numerical import difference, findiff, linneq
from numerical.isposdef import isposdef
from numerical.typedefs import ndarray
from overloads import bind_checker, dyn_typing
from overloads.shortcuts import assertNoInfNaN

from optimizer import pcg

Trust_Region_Format_T = Optional[
    Callable[
        [
            NamedArg(int, "iter"),
            NamedArg(float, "fval"),  # noqa: F821
            NamedArg(float, "step"),  # noqa: F821
            NamedArg(float, "grad"),  # noqa: F821
            NamedArg(int, "CGiter"),  # noqa: F821
            NamedArg(str, "CGexit"),  # noqa: F821
            NamedArg(str, "posdef"),  # noqa: F821
            NamedArg(Literal["Shaking", "       "], "shaking"),  # noqa: F821
        ],
        Optional[str],
    ]
]


class Grad_Check_Failed(BaseException):
    iter: int
    checker: Optional[Callable[[ndarray, ndarray], float]]
    analytic: ndarray
    findiff_: ndarray

    def __init__(
        self,
        iter: int,
        checker: Callable[[ndarray, ndarray], float],
        analytic: ndarray,
        findiff_: ndarray,
    ) -> None:
        self.iter = iter
        self.checker = checker
        self.analytic = analytic
        self.findiff_ = findiff_


_default_format_times: int = 0
_default_format_width: Dict[str, int] = {}


def default_format(
    *,
    iter: int,
    fval: float,
    step: float,
    grad: float,
    CGiter: int,
    CGexit: str,
    posdef: str,
    shaking: Literal["Shaking", "       "],
) -> str:
    global _default_format_width, _default_format_times
    data = {
        "Iter": f"{iter: 5d}",
        "F-Val": f"{fval: 10.8g}",
        "Step": f"{step:13.6g}",
        "Grad": f"{grad:6.4g}",
        "CG": f"{CGiter:2d}",
        "CG Exit": CGexit,
        "is Posdef": posdef,
        "Hessian": shaking,
    }
    output: List[str] = []
    for k, v in data.items():
        _width = _default_format_width.get(k, 0)
        _width = max(_width, max(len(k), len(v)))
        output.append(" " * (_width - len(v)) + v)
        _default_format_width[k] = _width
    _output = "  ".join(output)
    if _default_format_times % 20 == 0:
        label: List[str] = []
        for k in data:
            _width = _default_format_width[k]
            label.append(" " * (_width - len(k)) + k)
        _output = "\n" + "  ".join(label) + "\n\n" + _output
    _default_format_times += 1
    return _output


class Trust_Region_Options:
    border_abstol: Optional[float] = None
    tol_step: float = 1.0e-10
    tol_grad: float = 1.0e-6
    abstol_fval: Optional[float] = None
    max_stall_iter: Optional[int] = None
    init_delta: float = 1.0
    max_iter: int
    check_rel: float = 1.0e-2
    check_abs: Optional[float] = None
    check_iter: Optional[int] = None  # 0表示只在最优化开始前进行一次梯度检查，-1表示完全关闭检查，默认的None表示始终进行检查
    shaking: Union[Literal["x.shape[0]"], int] = "x.shape[0]"
    format: Trust_Region_Format_T
    posdef: Optional[Callable[[ndarray], str]]

    def __init__(self, *, max_iter: int) -> None:
        self.max_iter = max_iter
        self.format = default_format
        self.posdef = lambda H: "-*- ill -*-" if not isposdef(H) else ""


class Trust_Region_Result:
    x: ndarray
    iter: int
    delta: float
    gradient: ndarray
    success: bool

    def __init__(
        self, x: ndarray, iter: int, delta: float, grad: ndarray, *, success: bool
    ) -> None:
        self.x = x
        self.iter = iter
        self.delta = delta
        self.gradient = grad
        self.success = success


def _input_check(
    input: Tuple[
        Callable[[ndarray], float],
        Callable[[ndarray], ndarray],
        ndarray,
        ndarray,
        ndarray,
        ndarray,
        ndarray,
        Trust_Region_Options,
    ]
) -> None:
    _, _, x, A, b, lb, ub, _ = input
    linneq.constraint_check(A, b, lb, ub, theta=x)


def _output_check(output: Trust_Region_Result) -> None:
    assertNoInfNaN(output.x)


N = dyn_typing.SizeVar()
nConstraint = dyn_typing.SizeVar()


@dyn_typing.dyn_check_8(
    input=(
        dyn_typing.Callable(),
        dyn_typing.Callable(),
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.NDArray(numpy.float64, (nConstraint, N)),
        dyn_typing.NDArray(numpy.float64, (nConstraint,)),
        dyn_typing.NDArray(numpy.float64, (N,)),  # force line wrap
        dyn_typing.NDArray(numpy.float64, (N,)),
        dyn_typing.Class(Trust_Region_Options),
    ),
    output=dyn_typing.Class(Trust_Region_Result),
)
@bind_checker.bind_checker_8(input=_input_check, output=_output_check)
def trust_region(
    objective: Callable[[ndarray], float],
    gradient: Callable[[ndarray], ndarray],
    x: ndarray,
    constr_A: ndarray,
    constr_b: ndarray,
    constr_lb: ndarray,
    constr_ub: ndarray,
    opts: Trust_Region_Options,
) -> Trust_Region_Result:

    _hess_is_up_to_date: bool = False
    _hess_shaked: bool = False
    shaking: int = 0

    def objective_ndarray(x: ndarray) -> ndarray:
        return numpy.array([objective(x)])

    def grad_patch(x: ndarray, g: ndarray) -> ndarray:
        if opts.border_abstol is not None:
            g[numpy.logical_and(x - constr_lb < opts.border_abstol, g > 0)] = 0.0
            g[numpy.logical_and(constr_ub - x < opts.border_abstol, g < 0)] = 0.0
        return g

    def output(
        iter: int,
        fval: float,
        step_size: float,
        grad_infnorm: float,
        CGiter: Optional[int],
        CGexit: Optional[pcg.PCG_EXIT_FLAG],
        hessian: ndarray,
    ) -> None:
        nonlocal _hess_shaked
        if opts.format is not None:
            output = opts.format(
                iter=iter,
                fval=fval,
                step=step_size,
                grad=grad_infnorm,
                CGiter=CGiter if CGiter is not None else 0,
                CGexit=CGexit.name if CGexit is not None else "None",
                posdef=opts.posdef(hessian) if opts.posdef is not None else "",
                shaking="Shaking" if _hess_shaked else "       ",
            )
            if output is not None:
                print(output)
        _hess_shaked = False

    def make_grad(
        x: ndarray, iter: int, grad_infnorm: float, init_grad_infnorm: float
    ) -> ndarray:
        analytic = gradient(x)
        while True:
            if opts.check_abs is None:
                if opts.check_iter is not None and iter > opts.check_iter:
                    break
                if grad_infnorm < init_grad_infnorm * opts.check_rel:
                    break

            findiff_ = findiff.findiff(
                objective_ndarray, x, constr_A, constr_b, constr_lb, constr_ub
            )
            assert len(findiff_.shape) == 2 and findiff_.shape[0] == 1
            findiff_.shape = (findiff_.shape[1],)

            if difference.relative(analytic, findiff_) > opts.check_rel:
                raise Grad_Check_Failed(iter, difference.relative, analytic, findiff_)
            if opts.check_abs is not None:
                if difference.absolute(analytic, findiff_) > opts.check_abs:
                    raise Grad_Check_Failed(
                        iter, difference.absolute, analytic, findiff_
                    )
            break
        return grad_patch(x, analytic)

    def get_info(
        x: ndarray, iter: int, grad_infnorm: float, init_grad_infnorm: float
    ) -> Tuple[ndarray, float, Tuple[ndarray, ndarray, ndarray, ndarray]]:
        new_grad = make_grad(x, iter, grad_infnorm, init_grad_infnorm)
        grad_infnorm = numpy.max(numpy.abs(new_grad))
        constraints = (constr_A, constr_b - constr_A @ x, constr_lb - x, constr_ub - x)
        return new_grad, grad_infnorm, constraints

    def make_hess(x: ndarray) -> ndarray:
        nonlocal _hess_is_up_to_date, shaking, _hess_shaked
        assert not _hess_is_up_to_date
        H = findiff.findiff(
            lambda x: grad_patch(x, gradient(x)),
            x,
            constr_A,
            constr_b,
            constr_lb,
            constr_ub,
        )
        H = (H.T + H) / 2.0
        _hess_is_up_to_date, _hess_shaked = True, True
        shaking = x.shape[0] if opts.shaking == "x.shape[0]" else opts.shaking
        return H

    iter: int = 0
    delta: float = opts.init_delta
    assert linneq.check(x.reshape(-1, 1), constr_A, constr_b, constr_lb, constr_ub)

    fval: float
    grad: ndarray
    grad_infnorm: float
    H: ndarray
    constraints: Tuple[ndarray, ndarray, ndarray, ndarray]

    fval = objective(x)
    grad, grad_infnorm, constraints = get_info(x, iter, numpy.inf, 0.0)
    H = make_hess(x)
    output(iter, fval, numpy.nan, grad_infnorm, None, None, H)

    init_grad_infnorm = grad_infnorm
    old_fval, stall_iter = fval, 0
    while True:
        # 失败情形的截止条件放在最前是因为pcg失败时的continue会导致后面代码被跳过
        if delta < opts.tol_step:  # 信赖域太小
            return Trust_Region_Result(
                x, iter, delta, grad, success=False
            )  # pragma: no cover
        if iter > opts.max_iter:  # 迭代次数超过要求
            return Trust_Region_Result(
                x, iter, delta, grad, success=False
            )  # pragma: no cover

        if shaking <= 0 and not _hess_is_up_to_date:
            H = make_hess(x)

        # PCG
        step: Optional[ndarray]
        qpval: Optional[float]
        pcg_iter: int
        exit_flag: pcg.PCG_EXIT_FLAG
        step, qpval, pcg_iter, exit_flag = pcg.pcg(grad, H, constraints, delta)
        iter, shaking = iter + 1, shaking - 1

        if step is None:
            if _hess_is_up_to_date:
                delta /= 4.0
            else:
                H = make_hess(x)
            output(iter, fval, numpy.nan, grad_infnorm, pcg_iter, exit_flag, H)
            continue

        assert qpval is not None

        # 更新步长、试探点、试探函数值
        step_size: float = float(numpy.linalg.norm(step))  # type: ignore
        new_x: ndarray = x + step
        new_fval: float = objective(new_x)

        # 根据下降率确定信赖域缩放
        reduce: float = new_fval - fval
        ratio: float = 0 if reduce >= 0 else (1 if reduce <= qpval else reduce / qpval)
        if ratio >= 0.75 and step_size >= 0.9 * delta:
            delta *= 2
        elif ratio <= 0.25:
            if _hess_is_up_to_date:
                delta = step_size / 4.0
            else:
                H = make_hess(x)

        # 对符合下降要求的候选点进行更新
        if new_fval < fval:
            x, fval, _hess_is_up_to_date = new_x, new_fval, False
            grad, grad_infnorm, constraints = get_info(
                x, iter, grad_infnorm, init_grad_infnorm
            )
            if opts.abstol_fval is not None and old_fval - fval < opts.abstol_fval:
                stall_iter += 1
            else:
                old_fval, stall_iter = fval, 0

        output(iter, fval, step_size, grad_infnorm, pcg_iter, exit_flag, H)

        # 成功收敛准则
        if exit_flag == pcg.PCG_EXIT_FLAG.RESIDUAL_CONVERGENCE:  # PCG正定收敛
            if _hess_is_up_to_date:
                if grad_infnorm < opts.tol_grad:  # 梯度足够小
                    return Trust_Region_Result(x, iter, delta, grad, success=True)
                if step_size < opts.tol_step:  # 步长足够小
                    return Trust_Region_Result(x, iter, delta, grad, success=True)
            else:
                H = make_hess(x)

        if opts.max_stall_iter is not None and stall_iter >= opts.max_stall_iter:
            if _hess_is_up_to_date:
                return Trust_Region_Result(x, iter, delta, grad, success=True)
            else:
                H = make_hess(x)
