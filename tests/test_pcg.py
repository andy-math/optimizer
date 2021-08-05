import math

import numpy

from optimizer import pcg
from optimizer._internals.pcg.qpval import QuadEvaluator
from overloads.typedefs import ndarray

EPS = float(numpy.finfo(numpy.float64).eps)


class Test_pcg:
    def test_convex(self) -> None:
        numpy.random.seed(0)
        dim = 10
        delta = 9999
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
            qp_eval = QuadEvaluator(g=g, H=H)
            x = pcg._implimentation(qp_eval, delta)
            g = H @ x + g
            assert numpy.abs(g).max() < 10 * math.sqrt(EPS)

    def test_nonconvex(self) -> None:
        numpy.random.seed(0)
        dim = 10
        delta = 9999
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
            qp_eval = QuadEvaluator(g=g, H=H)
            x = pcg._implimentation(qp_eval, delta)
            x2: ndarray = numpy.linalg.lstsq(H, -g, rcond=None)[0]  # type: ignore
            assert 0.5 * (x @ H @ x) + g @ x <= 0.5 * (x2 @ H @ x2) + g @ x2


if __name__ == "__main__":
    Test_pcg().test_convex()
    Test_pcg().test_nonconvex()
