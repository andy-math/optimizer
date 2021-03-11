# Optimizer
[![Test](https://github.com/Andy-math/optimizer/actions/workflows/workflow.yaml/badge.svg)](https://github.com/Andy-math/optimizer/actions/workflows/workflow.yaml)
[![codecov](https://codecov.io/gh/Andy-math/optimizer/branch/main/graph/badge.svg?token=4GAZ3P5VX3)](https://codecov.io/gh/Andy-math/optimizer)

Python数值优化求解器

## Usage

### 无约束凸函数
`tests/test_trust_banana.py`
Rosenbrock's banana function: 
$$f\left(x\right) = 100\left(x_2 - x_1^2\right)^2 + \left(1-x_1\right)^2$$
$$J_f\left(x\right) = 
    \left(-400\left(x_2 - x_1^2\right)x_1 + 2\left(1-x_1\right), 200\left(x_2 - x_1^2\right)\right)$$
$$\hat{x} = \left(1, 1\right)$$
```python
from optimizer import trust_region


def func(_x: ndarray) -> float:
    x: float = float(_x[0])
    y: float = float(_x[1])
    return 100 * (y - x * x) ** 2 + (1 - x) ** 2


def grad(_x: ndarray) -> ndarray:
    x: float = float(_x[0])
    y: float = float(_x[1])
    return numpy.array([-400 * (y - x * x) * x - 2 * (1 - x), 200 * (y - x ** 2)])


n = 2
constr_A = numpy.zeros((0, n))
constr_b = numpy.zeros((0,))
constr_lb = numpy.full((n,), -numpy.inf)
constr_ub = numpy.full((n,), numpy.inf) 

opts = trust_region.Trust_Region_Options(max_iter=500)
result = trust_region.trust_region(
    func,
    grad,
    numpy.array([-1.9, 2]),
    constr_A,
    constr_b,
    constr_lb,
    constr_ub,
    opts,
)
print(result.success, result.x)
```
> \>\>\> True [0.99811542 0.99623087]

### 带约束非凸函数
`tests/test_trust_neg_curv.py`

$$f\left(x\right) = \frac{1}{x} + \ln{x}$$

$$f'\left(x\right) = \frac{1}{x} - \frac{1}{x^2}$$
$$x \in \left[0.25, 10\right]$$
$$\hat{x} = 1$$
```python
from optimizer import trust_region


def func(_x: ndarray) -> float:
    x: float = float(_x[0])
    return 1 / x + math.log(x)


def grad(_x: ndarray) -> ndarray:
    x: float = float(_x[0])
    return numpy.array([1 / x - 1 / (x * x)])


n = 1
constr_A = numpy.zeros((0, n))
constr_b = numpy.zeros((0,))
constr_lb = numpy.array([0.25])
constr_ub = numpy.array([10.0])

opts = trust_region.Trust_Region_Options(max_iter=500)
result = trust_region.trust_region(
    func,
    grad,
    numpy.array([9.5]),
    constr_A,
    constr_b,
    constr_lb,
    constr_ub,
    opts,
)
print(result.success, result.x)
```
> \>\>\> True [1.]