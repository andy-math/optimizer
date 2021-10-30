# Optimizer
[![Test](https://github.com/Andy-math/optimizer/actions/workflows/workflow.yaml/badge.svg)](https://github.com/Andy-math/optimizer/actions/workflows/workflow.yaml)
[![codecov](https://codecov.io/gh/Andy-math/optimizer/branch/main/graph/badge.svg?token=4GAZ3P5VX3)](https://codecov.io/gh/Andy-math/optimizer)

Python数值优化求解器

## Usage

### 无约束凸函数
`tests/test_trust_banana.py`

> Rosenbrock's banana function:
> 
> ![](https://latex.codecogs.com/gif.latex?f\\left\(x\\right\)=100\\left\(x_2-x_1^2\\right\)^2+\\left\(1-x_1\\right\)^2)
> 
> ![](https://latex.codecogs.com/gif.latex?J_f\\left\(x\\right\)=\\left\(-400\\left\(x_2-x_1^2\\right\)x_1-2\\left\(1-x_1\\right\),200\\left\(x_2-x_1^2\\right\)\\right\))
> 
> ![](https://latex.codecogs.com/gif.latex?\\hat{x}=\\left\(1,1\\right\))

```python
>>> import numpy
>>> from numpy import ndarray
>>> from optimizer import trust_region

>>> def func(_x: ndarray) -> float:
...     x: float = float(_x[0])
...     y: float = float(_x[1])
...     return 100 * (y - x * x) ** 2 + (1 - x) ** 2


>>> def grad(_x: ndarray) -> ndarray:
...     x: float = float(_x[0])
...     y: float = float(_x[1])
...     return numpy.array([-400 * (y - x * x) * x - 2 * (1 - x), 200 * (y - x ** 2)])


>>> n = 2
>>> constraints = (
...     numpy.zeros((0, n)), # A
...     numpy.zeros((0,)), # b
...     numpy.full((n,), -numpy.inf), # lb
...     numpy.full((n,), numpy.inf), # ub
... )

>>> opts = trust_region.Trust_Region_Options(max_iter=500)
>>> result = trust_region.trust_region(
...     func,
...     grad,
...     numpy.array([-1.9, 2]),
...     constraints,
...     opts,
... )
<BLANKLINE>
...Iter...
>>> print(result.success, result.x)
True [1. 1.]

```


### 带约束非凸函数
`tests/test_trust_neg_curv.py`

> ![](https://latex.codecogs.com/gif.latex?f\\left\(x\\right\)=\\frac{1}{x}+\\ln{x})
> 
> ![](https://latex.codecogs.com/gif.latex?f'\\left\(x\\right\)=\\frac{1}{x}-\\frac{1}{x^2})
> 
> ![](https://latex.codecogs.com/gif.latex?x\\in\\left[0.25,10\\right])
> 
> ![](https://latex.codecogs.com/gif.latex?\\hat{x}=1)

```python
>>> import math
>>> import numpy
>>> from numpy import ndarray
>>> from optimizer import trust_region

>>> def func(_x: ndarray) -> float:
...     x: float = float(_x[0])
...     return 1 / x + math.log(x)


>>> def grad(_x: ndarray) -> ndarray:
...     x: float = float(_x[0])
...     return numpy.array([1 / x - 1 / (x * x)])


>>> n = 1
>>> constraints = (
...     numpy.zeros((0, n)), # A
...     numpy.zeros((0,)), # b
...     numpy.array([0.25]), # lb
...     numpy.array([10.0]), # ub
... )

>>> opts = trust_region.Trust_Region_Options(max_iter=500)
>>> result = trust_region.trust_region(
...     func,
...     grad,
...     numpy.array([9.5]),
...     constraints,
...     opts,
... )
<BLANKLINE>
...Iter...
>>> print(result.success, result.x)
True [0.99999985]

```
