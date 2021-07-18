from typing import Optional


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
    check_iter: Optional[int] = None  # 0表示只在最优化开始前进行一次梯
    display: bool = True
    filename: str

    def __init__(self, *, max_iter: int, filename: str) -> None:
        self.max_iter = max_iter
        self.filename = filename
