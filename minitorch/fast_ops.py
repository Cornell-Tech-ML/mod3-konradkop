from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

from numba import prange
from numba import njit as _njit

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """A wrapper for the Numba `@njit` decorator with inlining set to "always". This
    function compiles the given function `fn` to machine code for faster execution
    using Numba's Just-In-Time (JIT) compilation. Additional keyword arguments can
    be passed to customize the JIT compilation.

    Args:
    ----
        fn (Fn): The function to be JIT-compiled.
        **kwargs (Any): Additional keyword arguments to pass to Numba's `@njit` decorator.

    Returns:
    -------
        Fn: The JIT-compiled version of the input function `fn`.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Iterate over each element in the output storage in parallel
        for index in prange(len(out)):
            # Create copies of the output and input shapes for index calculations
            outIndex = out_shape.copy()  # Temporary index for the output tensor
            inIndex = in_shape.copy()  # Temporary index for the input tensor

            # Convert the flat index `i` into a multi-dimensional index for the output
            to_index(index, out_shape, outIndex)

            # Adjust `outIndex` to match the broadcasting rules and get the corresponding input index
            broadcast_index(outIndex, out_shape, in_shape, inIndex)

            # Calculate the flat position in the output storage using the calculated output index
            outPos = index_to_position(outIndex, out_strides)

            # Calculate the flat position in the input storage using the broadcasted input index
            in_pos = index_to_position(inIndex, in_strides)

            # Apply the function `fn` to the input value and store the result in the output
            out[outPos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Iterate over each element in the output storage in parallel
        for index in prange(len(out)):
            # Create copies of shapes for temporary use in index calculations
            outIndex = out_shape.copy()  # Multi-dimensional index for the output tensor
            aIndex = a_shape.copy()  # Multi-dimensional index for input tensor A
            bIndex = b_shape.copy()  # Multi-dimensional index for input tensor B

            # Convert the flat index `i` into a multi-dimensional index for the output
            to_index(index, out_shape, outIndex)

            # Adjust `outIndex` to get corresponding indices for tensors A and B
            # Broadcast `outIndex` to match the shape of tensor A
            broadcast_index(outIndex, out_shape, a_shape, aIndex)
            # Broadcast `outIndex` to match the shape of tensor B
            broadcast_index(outIndex, out_shape, b_shape, bIndex)

            # Calculate the flat storage positions for tensors A and B using the broadcasted indices
            aPosition = index_to_position(aIndex, a_strides)
            bPosition = index_to_position(bIndex, b_strides)

            # Calculate the flat storage position for the output tensor using `outIndex`
            outPos = index_to_position(outIndex, out_strides)

            # Apply the function `fn` to the values from A and B, store the result in the output
            out[outPos] = fn(a_storage[aPosition], b_storage[bPosition])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for index in prange(len(out)):
            # Create copies of the shapes for index calculations
            out_index = (
                out_shape.copy()
            )  # Multi-dimensional index for the output tensor
            a_index = a_shape.copy()  # Multi-dimensional index for the input tensor

            # Convert the flat index `index` into a multi-dimensional index for the output
            to_index(index, out_shape, out_index)

            # Initialize `a_index` with the current `out_index`
            a_index = out_index.copy()

            # Iterate over the dimension being reduced
            for innerIndex in range(a_shape[reduce_dim]):
                # Adjust the current index in the reduction dimension
                a_index[reduce_dim] = innerIndex

                # Calculate the flat position in the input tensor using `a_index`
                a_pos = index_to_position(a_index, a_strides)

                # For the first iteration, initialize the output with the input value
                if innerIndex == 0:
                    out[index] = a_storage[a_pos]
                else:
                    # For subsequent iterations, apply the reduction function `fn`
                    out[index] = fn(out[index], a_storage[a_pos])
        # TODO: Implement for Task 3.1.
        # raise NotImplementedError("Need to implement for Task 3.1")

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # # TODO: Implement for Task 3.2.
    # raise NotImplementedError("Need to implement for Task 3.2")
    # Get shapes and dimensions for matrix multiplication
    batch_size = out_shape[0]  # Number of batches
    M = a_shape[-2]  # Rows in tensor A
    N = b_shape[-1]  # Columns in tensor B
    K = a_shape[-1]  # Inner dimension (shared by A and B)

    # Strides for batch dimension to handle broadcasting
    a_batch_stride = a_strides[0] if len(a_shape) > 2 and a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if len(b_shape) > 2 and b_shape[0] > 1 else 0
    out_batch_stride = out_strides[0] if len(out_shape) > 2 else 0

    # Iterate over each batch in parallel
    for batch in prange(batch_size):
        # Calculate base offsets for each batch
        a_batch_offset = batch * a_batch_stride
        b_batch_offset = batch * b_batch_stride
        out_batch_offset = batch * out_batch_stride

        # Iterate over rows of A and columns of B
        for m in range(M):
            for n in range(N):
                # Compute position in the output tensor
                out_pos = out_batch_offset + m * out_strides[-2] + n * out_strides[-1]

                # Initialize the output element to 0
                out[out_pos] = 0.0

                # Compute the dot product of the row of A and column of B
                for k in range(K):
                    # Calculate positions in A and B storage
                    a_pos = a_batch_offset + m * a_strides[-2] + k * a_strides[-1]
                    b_pos = b_batch_offset + k * b_strides[-2] + n * b_strides[-1]

                    # Perform the multiplication and accumulation
                    out[out_pos] += a_storage[a_pos] * b_storage[b_pos]


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
