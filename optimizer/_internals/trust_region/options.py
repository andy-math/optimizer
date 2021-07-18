import math
from typing import Callable, Dict, List, Literal, Optional, Union

from optimizer._internals.pcg.status import Status

_format_times: int = 0
_format_width: Dict[str, int] = {
    "Iter": 3,
    "F-Val": 14,
    "Step": 13,
    "Grad": 9,
    "CG": 1,
    "CG Exit": 20,
    "is Posdef": 11,
    "Hessian": 7,
}


def _format(
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
    global _format_width, _format_times
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
        _width = _format_width[k]
        _width = max(_width, max(len(k), len(v)))
        output.append(" " * (_width - len(v)) + v)
        _format_width[k] = _width
    _output = "  ".join(output)
    if iter == 0:
        _format_times = 0
    if _format_times % 20 == 0:
        label: List[str] = []
        for k in data:
            _width = _format_width[k]
            label.append(" " * (_width - len(k)) + k)
        _output = "\n" + "  ".join(label) + "\n\n" + _output
    _format_times += 1
    return _output


class Trust_Region_Options:
    border_abstol: float = 1e-10
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
    posdef: Optional[Callable[[bool], str]]
    display: bool = True

    def __init__(self, *, max_iter: int) -> None:
        self.max_iter = max_iter
        self.posdef = lambda ill: "-*- ill -*-" if ill else "           "


def format(
    iter: int,
    fval: float,
    grad_infnorm: float,
    pcg_status: Optional[Status],
    ill: bool,
    opts: Trust_Region_Options,
    times_after_hessian_shaking: int,
) -> str:
    assert times_after_hessian_shaking >= 1
    return _format(
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
        posdef="" if opts.posdef is None else opts.posdef(ill),
        shaking="Shaking" if times_after_hessian_shaking == 1 else "       ",
    )
