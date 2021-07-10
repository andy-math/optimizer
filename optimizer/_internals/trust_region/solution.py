# -*- coding: utf-8 -*-


from typing import Final, Tuple

from optimizer._internals.common.findiff import findiff
from optimizer._internals.common.hessian import Hessian
from optimizer._internals.trust_region.frozenstate import FrozenState
from optimizer._internals.trust_region.grad_maker import (
    Gradient,
    GradientCheck,
    make_gradient,
)
from overloads.typing import ndarray


class Solution:
    state: Final[FrozenState]
    fval: Final[float]
    x: Final[ndarray]
    grad: Final[Gradient]
    shifted_constr: Final[Tuple[ndarray, ndarray, ndarray, ndarray]]
    hess_up_to_date: bool = False

    # 对每一个当前的sol计算有效集A
    # 对A.T @ A分解得到0特征值列向量v, v是正交矩阵（的一些列）
    # 对从此处产生的grad, cut-off策略使用如下：
    # g = v @ [(g@v)/sum(v*v,axis=0)].T
    # 其中，norm2(v)^2 === 1 (单位化)，上式简化为
    # g = v @ (g@v).T
    # 进一步可以简化为
    # g = v @ v.T @ g.T
    # 因此，Type`ActiveSet`定义为[v @ v.T]
    # 当特征值全为0时，由正交矩阵，上式变为：
    # g = eye @ g
    # 当特征值全不为0时，定义[v @ v.T]为0，简记为None

    def __init__(
        self,
        iter: int,
        x: ndarray,
        g_infnorm: Tuple[(float, float)],
        state: FrozenState,
    ) -> None:
        self.state = state
        self.fval = state.f(x)
        self.x = x
        grad = make_gradient(
            state.g,
            x,
            state.constraints,
            state.opts,
            check=GradientCheck(state.f_np, iter, *g_infnorm),
        )
        self.grad = grad
        A, b, lb, ub = state.constraints
        self.shifted_constr = (A, b - A @ x, lb - x, ub - x)

    def get_hessian(self) -> Hessian:
        self.hess_up_to_date = True
        H = findiff(
            lambda x: make_gradient(
                self.state.g, x, self.state.constraints, self.state.opts, check=None
            ).value,
            self.x,
            self.state.constraints,
        )
        return Hessian(
            H,
            max_times=self.x.shape[0]
            if self.state.opts.shaking == "x.shape[0]"
            else self.state.opts.shaking,
        )
