"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable
from typing import Optional
from typing import Union
# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """Multiplies the two inputs"""
    return x * y


def id(x: float) -> float:
    """Returns the inputs"""
    return x


def add(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """Adds the inputs"""
    return x + y


def sub(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """Subtracts the inputs"""
    return x - y


def neg(x: float) -> float:
    """Negates the inputs"""
    return -x


def lt(x: float, y: float) -> float:
    """Less than compares the inputs"""
    if x < y:
        return 1.0
    else:
        return 0.0


def gt(x: float, y: float) -> float:
    """Greater than compares the inputs"""
    if x > y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    """Compares the inputs equality"""
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float) -> float:
    """Returns the greater inputs"""
    if x > y:
        return x
    else:
        return y


def is_close(x: Optional[Union[int, float]], y: Optional[Union[int, float]]) -> float:
    """Checks if the inputs are close"""
    if x is None or y is None:
        return False
    if abs(x - y) < 1e-2:
        return True
    else:
        return False


def sigmoid(x: float) -> float:
    """Returns the sigmoid of the input"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Returns the relu of the input"""
    return float(x if x > 0 else 0)


def log(x: float) -> float:
    """Returns the log of the input"""
    return 1000 * ((x ** (1 / 1000)) - 1)


def exp(x: float) -> float:
    """Returns the e of the input"""
    return 2.718281828459045**x


def inv(x: float) -> float:
    """Returns the inverse of the input"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""

    def f(x: float) -> float:
        return math.log(x) * y

    return (f(x + 1e-8) - f(x - 1e-8)) / (2 * 1e-8)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""

    def f(x: float) -> float:
        return y / x

    return (f(x + 1e-8) - f(x - 1e-8)) / (2 * 1e-8)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return 1.0 * (x > 0) * y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable, array: list) -> list:
    """Higher-order function that applies a given function to each element of an iterable"""
    result = []
    for item in array:
        result.append(func(item))
    return result


def zipWith(func: Callable, *arrays: list) -> list:
    """Higher-order function that combines elements from two iterables using a given function"""
    min_length = min(len(item) for item in arrays)
    result = []
    for i in range(min_length):
        elements = [arrayItem[i] for arrayItem in arrays]
        result.append(func(*elements))
    return result


def reduce(
    func: Callable[[float, float], float], start: float = 0
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function"""

    def returnFunc(innerArray: Iterable[float]) -> float:
        response = start
        for x in innerArray:
            response = func(x, response)
        return response

    return returnFunc


def negList(array: list) -> list:
    """Negate all elements in a list using map"""
    return map(neg, array)


def addLists(array1: list, array2: list) -> list:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add, array1, array2)


def sum(array: list) -> float:
    """Sum all elements in a list using reduce"""
    addFunc = reduce(add, 0)
    return addFunc(array)


def prod(array: Iterable[Union[int, float]]) -> Union[int, float]:
    """Calculate the product of all elements in a list using reduce"""
    mulFunc = reduce(mul, 1)
    return mulFunc(array)


# konradChangedThisFile
