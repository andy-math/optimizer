from typing import cast

import numpy
import scipy.stats  # type: ignore

from optimizer import trust_region
from overloads import difference
from overloads.typedefs import ndarray


class Sample:
    beta: ndarray
    X: ndarray
    Y: ndarray
    lambda_: float
    beta_decomp: ndarray

    def symm_eig(self, A: ndarray) -> ndarray:
        A = (A.T + A) / 2
        return cast(ndarray, numpy.linalg.eigh(A)[0])

    def orthogonal_X(self, X: ndarray) -> ndarray:
        for i in range(1, X.shape[1]):
            norm = numpy.sqrt(X[:, i] @ X[:, i])
            X[:, i] -= X[:, :i] @ numpy.linalg.lstsq(X[:, :i], X[:, i], rcond=None)[0]
            X[:, i] *= norm / numpy.sqrt(X[:, i] @ X[:, i])
        return X

    def soft_threshold(self, beta: ndarray, lambda_: ndarray) -> ndarray:
        beta = numpy.sign(beta) * numpy.maximum(numpy.abs(beta) - lambda_, 0.0)
        return beta

    def lasso_decomp(self) -> ndarray:
        m, n = self.X.shape
        beta_decomp: ndarray = numpy.linalg.lstsq(self.X, self.Y, rcond=None)[0]
        beta_decomp = self.soft_threshold(
            beta_decomp,
            (m / 2.0 * self.lambda_)
            * numpy.linalg.lstsq(self.X.T @ self.X, numpy.ones((n,)), rcond=None)[0],
        )
        return beta_decomp

    def __init__(self, m: int, n: int) -> None:
        self.beta = self.symm_eig(numpy.random.rand(n, n).T)
        self.X = self.orthogonal_X(scipy.stats.norm.ppf(numpy.random.rand(n, m).T))
        self.Y = self.X @ self.beta + scipy.stats.norm.ppf(numpy.random.rand(m))
        self.lambda_ = 2 * numpy.quantile(
            numpy.abs(numpy.linalg.lstsq(self.X, self.Y, rcond=None)[0]), 0.3
        )
        self.beta_decomp = self.lasso_decomp()


numpy.random.seed(5489)


def lasso_objective(beta: ndarray, X: ndarray, Y: ndarray, lambda_: float) -> float:
    err = Y - X @ beta
    obj = err @ err / err.shape[0]
    obj += lambda_ * numpy.sum(numpy.abs(beta))
    return float(obj)


def lasso_gradient(beta: ndarray, X: ndarray, Y: ndarray, lambda_: float) -> ndarray:
    err = Y - X @ beta
    grad = cast(ndarray, -(2 * err) @ X / err.shape[0])
    grad[beta < 0] += -lambda_
    grad[beta > 0] += lambda_
    return grad


def once(m: int, n: int) -> None:
    sample = Sample(m, n)

    constraints = (
        numpy.empty((0, n)),
        numpy.empty((0,)),
        numpy.full((n,), -numpy.inf),
        numpy.full((n,), numpy.inf),
    )

    opts = trust_region.Trust_Region_Options(max_iter=500)
    opts.check_rel = 1
    opts.check_abs = 1e-5
    result = trust_region.trust_region(
        lambda beta: lasso_objective(beta, sample.X, sample.Y, sample.lambda_),
        lambda beta: lasso_gradient(beta, sample.X, sample.Y, sample.lambda_),
        numpy.zeros((n,)),
        constraints,
        opts,
    )
    abserr = difference.absolute(result.x, sample.beta_decomp)
    relerr = abserr / numpy.mean(numpy.abs(sample.beta_decomp))
    print(f"result.success: {result.success}")
    print(f"relerr        : {relerr}")
    print(f"abserr        : {abserr}")
    print(
        "result.x      : \n",
        numpy.concatenate(
            (
                sample.beta_decomp.reshape((-1, 1)),
                result.x.reshape((-1, 1)),
            ),
            axis=1,
        ),
    )
    assert relerr < 0.35
    assert abserr < 0.35


class Test:
    def test1(self) -> None:
        once(m=1000, n=4)

    def test2(self) -> None:
        once(m=1000, n=8)

    def test3(self) -> None:
        once(m=1000, n=16)

    def test4(self) -> None:
        once(m=1000, n=32)


if __name__ == "__main__":
    Test().test1()
    Test().test2()
    Test().test3()
    Test().test4()
