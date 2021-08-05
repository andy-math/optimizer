import math
from typing import Dict, List, Literal, Optional

from optimizer._internals.common.hessian import Hessian
from optimizer._internals.quad_prog.status import Status
from optimizer._internals.trust_region.solution import Solution

_format_times: int = 0
_format_width: Dict[str, int] = {
    "Iter": 3,
    "F-Val": 14,
    "Step": 13,
    "Grad": 9,
    "Angle": 5,
    "Quad Prog": 10,
    "is Posdef": 11,
    "Hessian": 7,
}


def _format(
    *,
    iter: int,
    fval: float,
    step: float,
    grad: float,
    QPangle: float,
    QPexit: str,
    posdef: str,
    shaking: Literal["Shaking", "       "],
) -> str:
    global _format_width, _format_times
    data = {
        "Iter": f"{iter: 5d}",
        "F-Val": f"{fval: 10.8g}",
        "Step": f"{step:13.6g}",
        "Grad": f"{grad:6.4g}",
        "Angle": f"{QPangle:4.1f}%",
        "Quad Prog": QPexit,
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


def format(
    iter: int, sol: Solution, hessian: Hessian, pcg_status: Optional[Status]
) -> str:
    if _format_times == 0:
        assert hessian.times == 0
    else:
        assert hessian.times >= 1
    return _format(
        iter=iter,
        fval=sol.fval,
        step=(
            math.nan
            if pcg_status is None or pcg_status.size is None
            else pcg_status.size
        ),
        grad=sol.grad.infnorm,
        QPangle=0 if pcg_status is None else pcg_status.angle,
        QPexit="None" if pcg_status is None else pcg_status.flag.name,
        posdef="-*- ill -*-" if hessian.ill else "           ",
        shaking="Shaking" if hessian.times == 1 else "       ",
    )
