"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple, Optional

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


from typing import Any, Tuple, Optional, List


def wrap_tuple(x: Any) -> tuple:
    """Ensures the given input is a tuple. If the input is not already a tuple,
    it wraps the input inside a tuple.

    Args:
    ----
        x (Any): The input to be converted to a tuple.

    Returns:
    -------
        tuple: A tuple containing the input value, or the input itself if it was already a tuple.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Internal method to call the backward function and wrap the result in a tuple.

        Args:
        ----
            ctx (Context): The context holding saved data for the backward computation.
            grad_out (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tuple[Tensor, ...]: A tuple of gradients for each input tensor.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Internal method to call the forward function.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            *inps (Tensor): The input tensors to the forward function.

        Returns:
        -------
            Tensor: The result of the forward computation.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Applies the function by calling the forward function and tracking history if needed.

        Args:
        ----
            *vals (Tensor): The input tensors for the function.

        Returns:
        -------
            Tensor: The result of the forward computation with history tracking if necessary.

        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise negation of the input tensor.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor to negate.

        Returns:
        -------
            Tensor: A new tensor containing the negated values of `t1`.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the negation operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise inverse of the input tensor.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the inverse values of `t1`.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the inverse operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the element-wise addition of two tensors.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the sum of `t1` and `t2`.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of the addition operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to `t1` and `t2`.

        """
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the element-wise subtraction of two tensors.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The tensor from which to subtract.
            t2 (Tensor): The tensor to subtract.

        Returns:
        -------
            Tensor: A new tensor containing the result of `t1 - t2`.

        """
        return t1.f.sub_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of the subtraction operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to `t1` and `t2`.

        """
        return grad_output, Neg.apply(grad_output)


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor]) -> Tensor:
        """Checks if all elements are true across the specified dimension.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            a (Tensor): The input tensor.
            dim (Optional[Tensor]): The dimension along which to check.

        Returns:
        -------
            Tensor: A tensor containing the result of the logical operation.

        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the element-wise multiplication of two tensors.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the product of `a` and `b`.

        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of the multiplication operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to `a` and `b`.

        """
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(b, grad_output),
            grad_output.f.mul_zip(a, grad_output),
        )


class GT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the element-wise greater-than comparison between two tensors.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing boolean values indicating if `t1 > t2`.

        """
        return t1.f.gt_zip(t1, t2)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise sigmoid function of the input tensor.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the sigmoid values of `t1`.

        """
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the sigmoid operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (1.0 + (-sigma)) * grad_output


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the sum of tensor elements across a specified dimension.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            a (Tensor): The input tensor.
            dim (Tensor): The dimension along which to sum.

        Returns:
        -------
            Tensor: A new tensor containing the sum of elements along the specified dimension.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the gradient of the sum operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient with respect to the input tensor and a scalar.

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise ReLU (Rectified Linear Unit) of the input tensor.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the ReLU-activated values of `t1`.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the ReLU operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        (a,) = ctx.saved_values
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise natural logarithm of the input tensor.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the logarithm values of `t1`.

        """
        ctx.save_for_backward(t1)
        out = t1.f.log_map(t1)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the logarithm operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        (a,) = ctx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the element-wise exponential of the input tensor.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the exponential values of `t1`.

        """
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the gradient of the exponential operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tensor: The gradient with respect to the input tensor.

        """
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the element-wise less-than comparison between two tensors.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing boolean values indicating if `a < b`.

        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of the less-than comparison, which is always zero.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Zero gradients for both input tensors.

        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes the element-wise equality comparison between two tensors.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing boolean values indicating if `a == b`.

        """
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradient of the equality comparison, which is always zero.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Zero gradients for both input tensors.

        """
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Checks if two tensors are element-wise close within a tolerance.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor containing boolean values indicating if `a` is close to `b`.

        """
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permutes the dimensions of a tensor according to the given order.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            a (Tensor): The input tensor to permute.
            order (Tensor): The order in which to permute the dimensions.

        Returns:
        -------
            Tensor: A new tensor with permuted dimensions.

        """
        ctx.save_for_backward(order)
        return a._new(a._tensor.permute(*(int(order[i]) for i in range(order.size))))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the gradient of the permutation operation.

        Args:
        ----
            ctx (Context): The context that contains saved information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the loss function.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient with respect to the input tensor and a scalar.

        """
        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Performs a view operation on the input tensor, reshaping it to the specified dimensions.

        Args:
        ----
            ctx (Context): The context for saving information for backward computation.
            a (Tensor): The input tensor to be reshaped.
            shape (Tensor): A tensor representing the desired shape for the output.

        Returns:
        -------
            Tensor: A new tensor that shares the same storage as `a`, reshaped to the specified dimensions.

        Raises:
        ------
            AssertionError: If the input tensor is not contiguous.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes the gradient of a function using the central difference method.

    Args:
    ----
        f (Any): The function for which the gradient is being computed.
        *vals (Tensor): The input values for the function, with the gradient
                        being computed with respect to the `arg`-th value.
        arg (int, optional): The index of the value to differentiate. Defaults to 0.
        epsilon (float, optional): The small value used for the finite difference approximation. Defaults to 1e-6.
        ind (UserIndex): The index used to perturb the input value.

    Returns:
    -------
        float: The estimated gradient of the function with respect to the specified input.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
