from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Using Apply on the Scalar will give it a history"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward operation of addition.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: The result of adding a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradients for the inputs of the addition.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: Gradients with respect to the inputs.

        """
        return d_output, d_output


class Sub(ScalarFunction):
    """Subtract function $f(x, y) = x - y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward operation of subtraction.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The minuend.
            b (float): The subtrahend.

        Returns:
        -------
            float: The result of subtracting b from a.

        """
        return a - b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradients for the inputs of the subtraction.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: Gradients with respect to the inputs.

        """
        return d_output, -d_output


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward operation of multiplication.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: The result of multiplying a and b.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradients for the inputs of the multiplication.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: Gradients with respect to the inputs.

        """
        (
            a,
            b,
        ) = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = inv(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward operation of computing the inverse.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The inverse of a.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient for the input of the inverse function.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward operation of negation.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The negation of a.

        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient for the input of the negation function.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = sigmoid(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward operation of the sigmoid function.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the sigmoid function to a.

        """
        sig = operators.sigmoid(a)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient for the input of the sigmoid function.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (sig,) = ctx.saved_values
        return float((sig * (1 - sig)) * d_output)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = ReLU(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward operation of the ReLU function.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the ReLU function to a.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient for the input of the ReLU function.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward operation of the exponential function.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the exponential function to a.

        """
        ctx.save_for_backward(operators.exp(a))
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient for the input of the exponential function.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return a * d_output


class LT(ScalarFunction):
    """Less-than function :math:`f(x, y) = 1 (true)` true if x < y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward operation of the less-than comparison.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: 1.0 if a < b, otherwise 0.0.

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradients for the inputs of the less-than comparison.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to the inputs.

        """
        return 0.0, 0.0


class GT(ScalarFunction):
    """Greater-than function :math:`f(x, y) = 1 (true)` if x > y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward operation of the greater-than comparison.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: 1.0 if a > b, otherwise 0.0.

        """
        return operators.gt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradients for the inputs of the greater-than comparison.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to the inputs, which are both 0.0.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function :math:`f(x, y) = 1 (true) if x == y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward operation of the equality comparison.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The first operand.
            b (float): The second operand.

        Returns:
        -------
            float: 1.0 if a == b, otherwise 0.0.

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the gradients for the inputs of the equality comparison.

        Args:
        ----
            ctx (Context): The context object with saved values.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to the inputs, which are both 0.0.

        """
        return 0.0, 0.0


class Log(ScalarFunction):
    """Logarithm function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward operation of the logarithm function.

        Args:
        ----
            ctx (Context): The context object to save values for backward computation.
            a (float): The input value.

        Returns:
        -------
            float: The logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient for the input of the logarithm function.

        Args:
        ----
        ctx (Context): The context object with saved values.
        d_output (float): The gradient of the output.

        Returns:
        -------
        float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# konradChangedThisFile
