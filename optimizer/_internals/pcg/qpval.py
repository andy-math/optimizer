from numpy import ndarray
from overloads.shortcuts import assertNoInfNaN_float


def qpval(*, g: ndarray, H: ndarray, x: ndarray) -> float:
    assert len(g.shape) == 1
    assert len(H.shape) == 2
    assert len(x.shape) == 1
    assert g.shape[0] == H.shape[0] == H.shape[1] == x.shape[0]
    value = float(g @ x) + 0.5 * float(x @ H @ x)
    assertNoInfNaN_float(value)
    return value
