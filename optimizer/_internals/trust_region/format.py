from typing import Callable, Dict, List, Literal, Optional

from mypy_extensions import NamedArg

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
