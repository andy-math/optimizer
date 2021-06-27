import math
from typing import Callable, Dict, List, Literal, Optional, Union

from mypy_extensions import NamedArg
from numerical.isposdef import isposdef
from numerical.typedefs import ndarray
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
_default_format_times: int = 0
_default_format_width: Dict[str, int] = {
    "Iter": 3,
    "F-Val": 14,
    "Step": 13,
    "Grad": 9,
    "CG": 1,
    "CG Exit": 20,
    "is Posdef": 11,
    "Hessian": 7,
}


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
        _width = _default_format_width[k]
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
        self.posdef = lambda H: "-*- ill -*-" if not isposdef(H) else "           "


def output(
    iter: int,
    fval: float,
    grad_infnorm: float,
    pcg_status: Optional[pcg.PCG_Status],
    hessian: ndarray,
    opts: Trust_Region_Options,
    times_after_hessian_shaking: int,
) -> None:
    if opts.format is not None:
        output = opts.format(
            iter=iter,
            fval=fval,
            step=(
                math.nan
                if pcg_status is None or pcg_status.size is None
                else pcg_status.size
            ),
            grad=grad_infnorm,
            CGiter=0 if pcg_status is None else pcg_status.iter,
            CGexit="None" if pcg_status is None else pcg_status.flag.name,
            posdef="" if opts.posdef is None else opts.posdef(hessian),
            shaking="Shaking" if times_after_hessian_shaking == 1 else "       ",
        )
        if output is not None:
            print(output)
