from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    GT,
    Add,
    Sub,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        """Return a string representation of the Scalar object.

        Returns
        -------
            str: A string representation of the Scalar, showing its data.

        """
        return f"Scalar({self.data})"

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Perform true division of the Scalar by another Scalar or compatible type.

        Args:
        ----
            b (ScalarLike): The divisor.

        Returns:
        -------
            Scalar: The result of the division.

        """
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Perform true division of another Scalar or compatible type by this Scalar.

        Args:
        ----
            b (ScalarLike): The dividend.

        Returns:
        -------
            Scalar: The result of the division.

        """
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        """Return the truth value of the Scalar.

        Returns
        -------
            bool: True if the data is non-zero, otherwise False.

        """
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Perform addition of another Scalar or compatible type to this Scalar.

        Args:
        ----
            b (ScalarLike): The value to add.

        Returns:
        -------
            Scalar: The result of the addition.

        """
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Perform multiplication of another Scalar or compatible type with this Scalar.

        Args:
        ----
            b (ScalarLike): The value to multiply.

        Returns:
        -------
            Scalar: The result of the multiplication.

        """
        return self * b

    # konrad's stuff here:

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Perform multiplication of this Scalar by another Scalar or compatible type.

        Args:
        ----
            b (ScalarLike): The value to multiply.

        Returns:
        -------
            Scalar: The result of the multiplication.

        """
        return Mul.apply(self, b)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Perform addition of this Scalar and another Scalar or compatible type.

        Args:
        ----
            b (ScalarLike): The value to add.

        Returns:
        -------
            Scalar: The result of the addition.

        """
        return Add.apply(self, b)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Check if this Scalar is less than another Scalar or compatible type.

        Args:
        ----
            b (ScalarLike): The value to compare against.

        Returns:
        -------
            Scalar: The result of the less-than comparison.

        """
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Check if this Scalar is greater than another Scalar or compatible type.

        Args:
        ----
            b (ScalarLike): The value to compare against.

        Returns:
        -------
            Scalar: The result of the greater-than comparison.

        """
        return GT.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Perform subtraction of another Scalar or compatible type from this Scalar.

        Args:
        ----
            b (ScalarLike): The value to subtract.

        Returns:
        -------
            Scalar: The result of the subtraction.

        """
        return Sub.apply(self, b)

    def __neg__(self) -> Scalar:
        """Return the negation of this Scalar.

        Returns
        -------
            Scalar: The negated value of this Scalar.

        """
        return Neg.apply(self)

    def log(self) -> Scalar:
        """Compute the logarithm of this Scalar.

        Returns
        -------
            Scalar: The logarithm of the Scalar's value.

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Compute the exponential of this Scalar.

        Returns
        -------
            Scalar: The exponential of the Scalar's value.

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Compute the sigmoid of this Scalar.

        Returns
        -------
            Scalar: The sigmoid of the Scalar's value.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Compute the ReLU (Rectified Linear Unit) of this Scalar.

        Returns
        -------
            Scalar: The ReLU of the Scalar's value.

        """
        return ReLU.apply(self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Check if this Scalar is equal to another Scalar or compatible type.

        Args:
        ----
            b (ScalarLike): The value to compare against.

        Returns:
        -------
            Scalar: The result of the equality comparison.

        """
        return EQ.apply(self, b)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the current variable has any history"""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the previous inputs on the list of variables"""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Function should is able to backward process a function by passing it in a context and (d) and then collecting the local derivatives. It then pairs these with the right variables and returns them. This function is also where we filter out constants that were used on the forward pass, but do not need derivatives."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        inputs = h.inputs
        d = h.last_fn._backward(h.ctx, d_output)
        return zip(inputs, d)

        # TODO: Implement for Task 1.3.
        # raise NotImplementedError("Need to implement for Task 1.3")

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    # TODO: Implement for Task 1.2.
    #  raise NotImplementedError("Need to implement for Task 1.2")


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a Python function.

    Asserts False if the derivative is incorrect.

    Parameters
    ----------
    f : callable
        A function that takes n scalar inputs and returns a single scalar output.
    *scalars : Scalar
        The input scalar values for the function. These values are used to test
        the correctness of the automatic differentiation.

    Returns
    -------
    None

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )


# konradChangedThisFile
