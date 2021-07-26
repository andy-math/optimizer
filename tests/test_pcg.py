import math

import numpy
from optimizer import pcg
from overloads.typedefs import ndarray

EPS = float(numpy.finfo(numpy.float64).eps)


class Test_pcg:
    def test_convex(self) -> None:
        numpy.random.seed(0)
        dim = 10
        delta = 99999999
        constraints = (
            numpy.empty((0, dim)),
            numpy.empty((0,)),
            numpy.full((dim,), -numpy.inf),
            numpy.full((dim,), numpy.inf),
        )
        for _ in range(1000):
            H = numpy.random.randn(dim, dim)
            H = (H.T + H) / 2  # type: ignore
            V: ndarray
            E: ndarray
            E, V = numpy.linalg.eig(H)  # type: ignore
            E = E.real
            assert numpy.abs(
                V @ numpy.diag(E) @ V.T - H  # type: ignore
            ).max() < math.sqrt(EPS)
            H = (V * numpy.abs(E)) @ V.T

            H = (H.T + H) / 2  # type: ignore
            g = numpy.random.randn(dim)
            status, dir = pcg._implimentation(g, H, constraints, delta)
            assert status.flag == pcg.Flag.RESIDUAL_CONVERGENCE
            assert status.iter < dim
            assert dir is None
            g = H @ status.x + g
            assert numpy.abs(g).max() < math.sqrt(EPS)

    def test_nonconvex(self) -> None:
        numpy.random.seed(0)
        dim = 10
        delta = 99999999
        constraints = (
            numpy.empty((0, dim)),
            numpy.empty((0,)),
            numpy.full((dim,), -numpy.inf),
            numpy.full((dim,), numpy.inf),
        )
        for _ in range(1000):
            H = numpy.random.randn(dim, dim)
            H = (H.T + H) / 2  # type: ignore
            V: ndarray
            E: ndarray
            E, V = numpy.linalg.eig(H)  # type: ignore
            E = E.real
            assert numpy.abs(
                V @ numpy.diag(E) @ V.T - H  # type: ignore
            ).max() < math.sqrt(EPS)
            H = (V * numpy.random.randn(dim)) @ V.T

            H = (H.T + H) / 2  # type: ignore

            g = numpy.random.randn(dim)
            status, dir = pcg._implimentation(g, H, constraints, delta)
            if status.flag == pcg.Flag.RESIDUAL_CONVERGENCE:
                assert dir is None
                g = H @ status.x + g
                assert numpy.abs(g).max() < math.sqrt(EPS)
            else:
                assert status.flag == pcg.Flag.NEGATIVE_CURVATURE
                assert status.iter < dim
                assert dir is not None
                assert dir @ H @ dir < math.sqrt(EPS)


if __name__ == "__main__":
    Test_pcg().test_convex()
    Test_pcg().test_nonconvex()
