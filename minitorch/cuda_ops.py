# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Calculate the unique global index of the current thread
        # idx is the 1D index calculated from the block and thread indices
        global_index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # If the global index exceeds the output size, exit the function
        if global_index >= out_size:
            return

        # Initialize arrays to store the index positions for output and input tensors
        # These are local arrays to hold the multi-dimensional indices
        output_index = cuda.local.array(MAX_DIMS, numba.int32)
        input_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert the 1D global index to a multi-dimensional index for the output tensor
        to_index(global_index, out_shape, output_index)

        # Broadcast the output multi-dimensional index to the input shape
        # This adjusts the output index to match the corresponding input index
        broadcast_index(output_index, out_shape, in_shape, input_index)

        # Calculate the flat (1D) position in the output storage from the multi-dimensional output index
        output_position = index_to_position(output_index, out_strides)

        # Calculate the flat (1D) position in the input storage from the multi-dimensional input index
        input_position = index_to_position(input_index, in_strides)

        # Apply the function `fn` to the input value and store the result in the output
        out[output_position] = fn(in_storage[input_position])

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # out_index = cuda.local.array(MAX_DIMS, numba.int32)
        # a_index = cuda.local.array(MAX_DIMS, numba.int32)
        # b_index = cuda.local.array(MAX_DIMS, numba.int32)
        # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx >= out_size:
            return

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        to_index(idx, out_shape, out_index)

        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

        # # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_zip)  # type: ignore


# def _sum_practice(out: Storage, a: Storage, size: int) -> None:
#     """This is a practice sum kernel to prepare for reduce.

#     Given an array of length $n$ and out of size $n // \text{blockDIM}$
#     it should sum up each blockDim values into an out cell.

#     $[a_1, a_2, ..., a_{100}]$

#     |

#     $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

#     Note: Each block must do the sum using shared memory!

#     Args:
#     ----
#         out (Storage): storage for `out` tensor.
#         a (Storage): storage for `a` tensor.
#         size (int):  length of a.

#     """
#     BLOCK_DIM = 32

#     # cache = cuda.shared.array(BLOCK_DIM, numba.float64)
#     # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     # pos = cuda.threadIdx.x
#     local_idx = cuda.threadIdx.x
#     block_idx = cuda.blockIdx.x
#     shared_block = cuda.shared.array(BLOCK_DIM, numba.float64)
#     offset = 1
#     if block_idx * THREADS_PER_BLOCK + local_idx < size:
#         shared_block[local_idx] = a[block_idx * THREADS_PER_BLOCK + local_idx]
#     else:
#         shared_block[local_idx] = 0
#     while offset < BLOCK_DIM:
#         cuda.syncthreads()
#         if local_idx % (offset * 2) == 0:
#             shared_block[local_idx] += shared_block[local_idx + offset]
#         offset *= 2
#     out[block_idx] = shared_block[0]

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for a reduction operation.

    Given an input array `a` of length `size`, the goal is to sum up each block
    of `BLOCK_DIM` elements into a single cell in the `out` array. The length 
    of the `out` array should be `size // BLOCK_DIM`, since each block of `BLOCK_DIM`
    values in `a` will correspond to a single output value in `out`.

    The input array `a` is conceptually divided into chunks (or "blocks") of `BLOCK_DIM` elements:
    
        Input:  [a_1, a_2, ..., a_{size}]
        Output: [sum(a_1 ... a_{BLOCK_DIM}), sum(a_{BLOCK_DIM+1} ... a_{2*BLOCK_DIM}), ...]

    Note:
    -----
    Each thread block will be responsible for computing the sum of `BLOCK_DIM` elements using
    shared memory. This ensures efficient use of memory and parallel computation within each block.

    Args:
    ----
        out (Storage): Storage object where the reduced output is stored.
        a (Storage): Storage object containing the input data to be summed.
        size (int): The length of the input array `a`.
    """

    # Define the size of each block to be summed
    BLOCK_DIM = 32

    # Allocate shared memory for the current block. This shared memory is visible 
    # to all threads in the same block and can be used for communication and temporary 
    # storage of intermediate results within a block.
    shared_block = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Compute the local thread index within the block and the global block index.
    # `local_idx` is the index of the current thread within the block.
    # `block_idx` is the index of the block within the grid of blocks.
    local_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x

    # Calculate the offset position to access the correct element in the input array `a`.
    # Each block handles a segment of the input array `a` based on the block index and thread index.
    input_index = block_idx * BLOCK_DIM + local_idx

    # Initialize the shared memory for the current block with elements from the input array.
    # If the input_index is out of bounds (i.e., beyond the size of the input array), 
    # fill that position in the shared memory with 0.
    if input_index < size:
        shared_block[local_idx] = a[input_index]
    else:
        shared_block[local_idx] = 0

    # Synchronize all threads in the block to ensure that the shared memory has been
    # completely populated before performing any further operations.
    cuda.syncthreads()

    # Use a reduction pattern to sum up the elements in the shared memory.
    # The `offset` variable starts at 1 and doubles each iteration, controlling
    # the distance between elements being summed.
    offset = 1
    while offset < BLOCK_DIM:
        # Synchronize threads before performing each step to ensure all previous
        # updates to shared memory are visible to every thread.
        cuda.syncthreads()

        # Only threads whose index is a multiple of `2 * offset` participate in the summing.
        # They add the value at their position to the value `offset` positions away.
        if local_idx % (offset * 2) == 0:
            shared_block[local_idx] += shared_block[local_idx + offset]

        # Double the offset to continue reducing the number of active threads.
        offset *= 2

    # After the reduction is complete, the sum for this block is stored in the first
    # position of the shared memory (`shared_block[0]`). This value is then written
    # to the output array at the position corresponding to the current block.
    if local_idx == 0:
        out[block_idx] = shared_block[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


# def tensor_reduce(
#     fn: Callable[[float, float], float],
# ) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
#     """CUDA higher-order tensor reduce function.

#     Args:
#     ----
#         fn: reduction function maps two floats to float.

#     Returns:
#     -------
#         Tensor reduce function.

#     """

#     def _reduce(
#         out: Storage,
#         out_shape: Shape,
#         out_strides: Strides,
#         out_size: int,
#         a_storage: Storage,
#         a_shape: Shape,
#         a_strides: Strides,
#         reduce_dim: int,
#         reduce_value: float,
#     ) -> None:
#         BLOCK_DIM = 1024
#         # cache = cuda.shared.array(BLOCK_DIM, numba.float64)
#         # out_index = cuda.local.array(MAX_DIMS, numba.int32)
#         # out_pos = cuda.blockIdx.x
#         # pos = cuda.threadIdx.x
#         reduce_size = a_shape[reduce_dim]
#         local_idx = cuda.threadIdx.x
#         block_idx = cuda.blockIdx.x
#         shared_block = cuda.shared.array(BLOCK_DIM, numba.float64)
#         offset = 1
#         out_index = cuda.local.array(MAX_DIMS, numba.int32)
#         to_index(block_idx, out_shape, out_index)
#         out_position = index_to_position(out_index, out_strides)
#         if local_idx < reduce_size:
#             out_index[reduce_dim] = local_idx
#             shared_block[local_idx] = a_storage[index_to_position(out_index, a_strides)]
#         else:
#             shared_block[local_idx] = reduce_value
#         while offset < BLOCK_DIM:
#             cuda.syncthreads()
#             if local_idx % (offset * 2) == 0:
#                 shared_block[local_idx] = fn(
#                     shared_block[local_idx], shared_block[local_idx + offset]
#                 )
#             offset *= 2
#         cuda.syncthreads()
#         if local_idx == 0:
#             out[out_position] = shared_block[local_idx]

#     return jit(_reduce)  # type: ignore
def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Generates a CUDA kernel for performing a reduction operation on a tensor.

    This function is a higher-order function that takes a binary reduction function
    (such as addition or maximum) and returns a CUDA kernel that performs a reduction
    along a specified dimension of a tensor. This allows for operations like summing 
    over a dimension, finding the maximum, etc., to be easily implemented in parallel
    using CUDA.

    Args:
    ----
        fn (Callable[[float, float], float]): A binary function that takes two floats
            and returns a float. This function defines the reduction operation to be 
            performed, e.g., summing two floats or finding the maximum of two floats.

    Returns:
    -------
        Callable: A CUDA kernel function that performs the reduction. The returned
        function takes several parameters, including the input and output storage,
        tensor shapes, strides, the dimension to reduce, and the initial reduction value.
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        """
        The CUDA kernel function for performing the tensor reduction along a specified dimension.

        This function uses a tree-based reduction approach, leveraging shared memory for efficient 
        parallel computation within a block. Each thread block reduces a segment of the input tensor
        along the specified `reduce_dim`, and writes the result to the `out` tensor.

        Args:
        ----
            out (Storage): The storage object where the reduced result is stored.
            out_shape (Shape): The shape of the output tensor after reduction.
            out_strides (Strides): The strides of the output tensor.
            out_size (int): The total number of elements in the output tensor.
            a_storage (Storage): The storage object for the input tensor.
            a_shape (Shape): The shape of the input tensor.
            a_strides (Strides): The strides of the input tensor.
            reduce_dim (int): The dimension along which the reduction is performed.
            reduce_value (float): The initial value for the reduction (e.g., 0 for sum).
        """

        # Define the size of each thread block (number of threads in a block).
        # Each block will handle a segment of the reduction task.
        BLOCK_DIM = 1024

        # Compute the size of the dimension being reduced, indicating how many elements 
        # are to be reduced in each block.
        reduce_size = a_shape[reduce_dim]

        # Calculate the thread's local index within the block and the block's index within the grid.
        # `local_idx` identifies the position of the thread within the block.
        # `block_idx` specifies the position of the block within the grid of blocks.
        local_idx = cuda.threadIdx.x
        block_idx = cuda.blockIdx.x

        # Shared memory allocation for the current block. Shared memory is used for fast communication 
        # between threads in the same block and for storing intermediate results during the reduction.
        shared_block = cuda.shared.array(BLOCK_DIM, numba.float64)

        # Allocate a local array to store the multi-dimensional output index temporarily.
        # This array helps track the current indices within the output tensor.
        out_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert the current block index into a multi-dimensional index based on the output shape.
        # This conversion helps in navigating the tensor's multi-dimensional space.
        to_index(block_idx, out_shape, out_index)

        # Calculate the linear position in the output tensor's flattened storage using strides.
        # The result indicates where the final reduced value should be stored in `out`.
        out_position = index_to_position(out_index, out_strides)

        # Populate the shared memory with elements from the input tensor.
        # Each thread loads its corresponding value from the input tensor.
        if local_idx < reduce_size:
            # Adjust the out_index to point to the specific slice being reduced.
            out_index[reduce_dim] = local_idx
            # Convert the updated multi-dimensional index to a linear position in the input storage.
            input_position = index_to_position(out_index, a_strides)
            # Load the value from the input tensor into the shared memory.
            shared_block[local_idx] = a_storage[input_position]
        else:
            # If the thread's local index exceeds the size of the dimension being reduced,
            # initialize that position in shared memory with the `reduce_value`.
            shared_block[local_idx] = reduce_value

        # Synchronize all threads in the block to ensure shared memory is correctly populated
        # before performing the reduction operation.
        cuda.syncthreads()

        # Perform the reduction using a hierarchical tree-based approach.
        # The reduction happens in several steps, gradually reducing the number of active threads.
        offset = 1
        while offset < BLOCK_DIM:
            # Synchronize threads to ensure all threads have up-to-date data in shared memory
            # before proceeding to the next step.
            cuda.syncthreads()

            # Only threads whose index is a multiple of `2 * offset` will participate in this step.
            # These threads reduce their current value with the value located `offset` positions away.
            if local_idx % (offset * 2) == 0:
                shared_block[local_idx] = fn(
                    shared_block[local_idx], shared_block[local_idx + offset]
                )

            # Double the offset for the next reduction step.
            offset *= 2

        # A final synchronization to ensure that the reduction process is complete for all threads.
        cuda.syncthreads()

        # The final reduced result for this block is now in `shared_block[0]`.
        # The first thread in each block writes this result to the correct position in the output storage.
        if local_idx == 0:
            out[out_position] = shared_block[0]

    # Return the compiled CUDA kernel for the reduction operation.
    # `jit` is used to compile `_reduce` into a GPU kernel for execution.
    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    shm_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shm_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    idx_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idx_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if idx_x >= size or idx_y >= size:
        return
    
    pos = index_to_position((idx_x, idx_y), (size, 1))
    shm_a[idx_x][idx_y] = a[pos]
    shm_b[idx_x][idx_y] = b[pos]

    cuda.syncthreads()

    total = 0.0
    for i in range(size):
        total += shm_a[idx_x][i] * shm_b[i][idx_y]
    
    out[pos] = total


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


# def _tensor_matrix_multiply(
#     out: Storage,
#     out_shape: Shape,
#     out_strides: Strides,
#     out_size: int,
#     a_storage: Storage,
#     a_shape: Shape,
#     a_strides: Strides,
#     b_storage: Storage,
#     b_shape: Shape,
#     b_strides: Strides,
# ) -> None:
#     """CUDA tensor matrix multiply function.

#     Requirements:

#     * All data must be first moved to shared memory.
#     * Only read each cell in `a` and `b` once.
#     * Only write to global memory once per kernel.

#     Should work for any tensor shapes that broadcast as long as ::

#     ```python
#     assert a_shape[-1] == b_shape[-2]
#     ```
#     Returns:
#         None : Fills in `out`
#     """
#     a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
#     b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
#     # Batch dimension - fixed
#     batch = cuda.blockIdx.z

#     BLOCK_DIM = 32
#     # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
#     # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

#     # # The final position c[i, j]
#     # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     # j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

#     # # The local position in the block.
#     # pi = cuda.threadIdx.x
#     # pj = cuda.threadIdx.y

#     # # Code Plan:
#     # # 1) Move across shared dimension by block dim.
#     # #    a) Copy into shared memory for a matrix.
#     # #    b) Copy into shared memory for b matrix
#     # #    c) Compute the dot produce for position c[i, j]
#     # # TODO: Implement for Task 3.4.
#     # raise NotImplementedError("Need to implement for Task 3.4")


# tensor_matrix_multiply = jit(_tensor_matrix_multiply)
import numba
from numba import cuda

def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    BLOCK_DIM = 32

    # Allocate shared memory for tiles of A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Calculate indices for threads
    batch = cuda.blockIdx.z  # Batch dimension if present
    i = cuda.blockIdx.x * BLOCK_DIM + cuda.threadIdx.x  # Row index in C
    j = cuda.blockIdx.y * BLOCK_DIM + cuda.threadIdx.y  # Column index in C

    # Indices within the block
    local_x = cuda.threadIdx.x
    local_y = cuda.threadIdx.y

    # Initialize the accumulator for the dot product
    c_value = 0.0

    # Loop over tiles in the shared dimension
    for k in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load the current tile from matrix A into shared memory
        if i < a_shape[-2] and k * BLOCK_DIM + local_y < a_shape[-1]:
            a_index = (batch * a_strides[0] + i * a_strides[-2] + (k * BLOCK_DIM + local_y) * a_strides[-1])
            a_shared[local_x, local_y] = a_storage[a_index]
        else:
            a_shared[local_x, local_y] = 0.0  # Padding with zero if out of bounds

        # Load the current tile from matrix B into shared memory
        if k * BLOCK_DIM + local_x < b_shape[-2] and j < b_shape[-1]:
            b_index = (batch * b_strides[0] + (k * BLOCK_DIM + local_x) * b_strides[-2] + j * b_strides[-1])
            b_shared[local_x, local_y] = b_storage[b_index]
        else:
            b_shared[local_x, local_y] = 0.0  # Padding with zero if out of bounds

        # Synchronize threads to ensure the tile is fully loaded
        cuda.syncthreads()

        # Perform the dot product for the current tile
        for n in range(BLOCK_DIM):
            c_value += a_shared[local_x, n] * b_shared[n, local_y]

        # Synchronize again to ensure the previous calculation is complete before loading new tiles
        cuda.syncthreads()

    # Write the result to global memory
    if i < out_shape[-2] and j < out_shape[-1]:
        out_index = (batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1])
        out[out_index] = c_value

tensor_matrix_multiply = jit(_tensor_matrix_multiply)  # type: ignore
