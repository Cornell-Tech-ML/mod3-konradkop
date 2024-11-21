"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets the gradient tracking requirement for the tensor.

        Args:
        ----
            x (bool): If True, enables gradient tracking. If False, disables it.

        This method initializes a history for tracking operations
        if gradient tracking is enabled.

        """
        self.history = History()

    def zero_grad_(self) -> None:
        """Resets the gradient of the tensor to None.

        This method is typically called before performing a new backward
        pass to ensure that gradients from previous iterations do not accumulate.
        """
        self.grad = None

    def requires_grad(self) -> bool:
        """Checks if gradient tracking is enabled for the tensor.

        Returns
        -------
            bool: True if gradient tracking is enabled, False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros.

        Args:
        ----
            shape (Optional[UserShape]): The shape of the tensor to create.
                If None, the tensor will have the same shape as the instance.

        Returns:
        -------
            Tensor: A new tensor initialized with zeros and the specified shape.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is constant (i.e., does not require gradients).

        Returns
        -------
            bool: True if the tensor is constant, False otherwise.

        A tensor is considered constant if it has no history of operations
        that would require gradient computation.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Retrieves the input variables (parents) of the current variable.

        Returns
        -------
            Iterable[Variable]: An iterable of input variables that contributed to
            the current variable's value.

        Raises
        ------
            AssertionError: If the variable's history is not set, indicating that
            there are no parent variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for backpropagation.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to the loss.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: A generator yielding tuples of input variables
            and their corresponding gradients.

        Raises:
        ------
            AssertionError: If the history or required attributes are not present, or if
            the number of gradients does not match the number of inputs.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation to compute gradients.

        Args:
        ----
            grad_output (Optional[Tensor]): The gradient of the output with respect
            to the loss. If None, defaults to a tensor of ones with the same shape
            as the current tensor (must be a scalar if shape is not (1,)).

        Raises:
        ------
            AssertionError: If grad_output is None and the tensor is not a scalar.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Performs element-wise division of the current tensor by another tensor.

        Args:
        ----
            b (TensorLike): The tensor to divide by.

        Returns:
        -------
            Tensor: A new tensor resulting from the element-wise division.

        """
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Performs element-wise division of another tensor by the current tensor.

        Args:
        ----
            b (TensorLike): The tensor to be divided by the current tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from the element-wise division.

        """
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    # Functions
    # TODO: Implement for Task 2.3.
    # konrad work starts here
    @property
    def dims(self) -> int:
        """Returns
        int : dimensionality of the tensor

        """
        return self._tensor.dims

    @property
    def size(self) -> int:
        """Returns
        int : size of the tensor

        """
        return self._tensor.size

    # Functions
    def __add__(self, b: TensorLike) -> Tensor:
        """Adds the current tensor with another tensor or tensor-like object.

        Args:
        ----
            b (TensorLike): The tensor or tensor-like object to be added.

        Returns:
        -------
            Tensor: A new tensor that is the result of the element-wise addition.

        """
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        """Subtracts another tensor or tensor-like object from the current tensor.

        Args:
        ----
            b (TensorLike): The tensor or tensor-like object to be subtracted.

        Returns:
        -------
            Tensor: A new tensor that is the result of the element-wise subtraction.

        """
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        """Multiplies the current tensor with another tensor or tensor-like object.

        Args:
        ----
            b (TensorLike): The tensor or tensor-like object to be multiplied.

        Returns:
        -------
            Tensor: A new tensor that is the result of the element-wise multiplication.

        """
        return Mul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b: TensorLike) -> Tensor:
        """Compares each element of the current tensor to another tensor or tensor-like
        object and returns a tensor of boolean values indicating if the current
        tensor is less than the other.

        Args:
        ----
            b (TensorLike): The tensor or tensor-like object to compare.

        Returns:
        -------
            Tensor: A tensor of boolean values from the comparison.

        """
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:
        """Compares each element of the current tensor to another tensor or tensor-like
        object and returns a tensor of boolean values indicating if the current
        tensor is equal to the other.

        Args:
        ----
            b (TensorLike): The tensor or tensor-like object to compare.

        Returns:
        -------
            Tensor: A tensor of boolean values from the comparison.

        """
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        """Compares each element of the current tensor to another tensor or tensor-like
        object and returns a tensor of boolean values indicating if the current
        tensor is greater than the other.

        Args:
        ----
            b (TensorLike): The tensor or tensor-like object to compare.

        Returns:
        -------
            Tensor: A tensor of boolean values from the comparison.

        """
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        """Negates each element of the current tensor.

        Returns
        -------
            Tensor: A new tensor with each element negated.

        """
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        """Right-side addition, adds another tensor or tensor-like object to the current
        tensor.

        Args:
        ----
            b (TensorLike): The tensor or tensor-like object to be added.

        Returns:
        -------
            Tensor: A new tensor that is the result of the element-wise addition.

        """
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        """Right-side multiplication, multiplies another tensor or tensor-like object
        with the current tensor.

        Args:
        ----
            b (TensorLike): The tensor or tensor-like object to be multiplied.

        Returns:
        -------
            Tensor: A new tensor that is the result of the element-wise multiplication.

        """
        return self * b

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Checks if all elements in the tensor are true along a given dimension.

        Args:
        ----
            dim (Optional[int], optional): The dimension to check along. If None, checks
            all elements in the flattened tensor.

        Returns:
        -------
            Tensor: A tensor of boolean values indicating the result.

        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        """Compares each element of the current tensor to another tensor element-wise
        and returns a tensor of boolean values indicating if they are close to each
        other within a certain tolerance.

        Args:
        ----
            y (Tensor): The tensor to compare with.

        Returns:
        -------
            Tensor: A tensor of boolean values indicating if elements are close.

        """
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid function element-wise to the current tensor.

        Returns
        -------
            Tensor: A new tensor with the sigmoid function applied to each element.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU (Rectified Linear Unit) function element-wise to the current
        tensor, setting negative values to zero.

        Returns
        -------
            Tensor: A new tensor with the ReLU function applied to each element.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Applies the natural logarithm function element-wise to the current tensor.

        Returns
        -------
            Tensor: A new tensor with the natural logarithm applied to each element.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Applies the exponential function element-wise to the current tensor.

        Returns
        -------
            Tensor: A new tensor with the exponential function applied to each element.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Compute the sum over dimension `dim`"""
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean over dimension `dim`"""
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Permute tensor dimensions to *order"""
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Change the shape of the tensor to a new shape with the same size"""
        return View.apply(self, tensor(list(shape)))

    def zero_grad(self) -> None:  # pragma: no cover
        """Reset the derivative on this variable."""
        self.grad = None
