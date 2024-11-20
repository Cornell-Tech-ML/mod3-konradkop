# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba import prange
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

        # Calculate the unique global index of the current thread
        global_index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # If the global index exceeds the output size, exit the function
        if global_index >= out_size:
            return

        # Initialize arrays to store the index positions for output and input tensors
        output_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Convert the 1D global index to a multi-dimensional index for the output tensor
        to_index(global_index, out_shape, output_index)

        # Broadcast the output multi-dimensional index to both input shapes
        # This adjusts the output index to match the corresponding input indices
        broadcast_index(output_index, out_shape, a_shape, a_index)
        broadcast_index(output_index, out_shape, b_shape, b_index)

        # Calculate the flat (1D) position in the output storage from the multi-dimensional output index
        output_position = index_to_position(output_index, out_strides)

        # Calculate the flat (1D) positions in the input storage from the multi-dimensional indices
        a_position = index_to_position(a_index, a_strides)
        b_position = index_to_position(b_index, b_strides)

        # Apply the function `fn` to the elements from the two input tensors and store the result in the output
        out[output_position] = fn(a_storage[a_position], b_storage[b_position])

        # # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    CUDA kernel to compute the block-wise sum of elements in an input array.

    This kernel divides the input array into blocks of size `BLOCK_DIM` and computes
    the sum of each block's elements. The result is stored in the corresponding position
    in the output array. Shared memory is used to store intermediate sums within each block.

    Example:
        Input array: [a_1, a_2, ..., a_{100}]
        Output array: [sum(a_1 to a_{31}), sum(a_{32} to a_{63}), ...]

    Note: The kernel handles cases where the input size is not a multiple of `BLOCK_DIM`
    by initializing shared memory with zero for out-of-bounds threads.

    Args:
    ----
        out (Storage): Output storage to store the block-wise sums.
        a (Storage): Input storage containing the elements to sum.
        size (int): The total number of elements in the input array.
    """
    # Define the number of threads per block
    BLOCK_DIM = 32

    # Thread index within the block (local index)
    local_thread_idx = cuda.threadIdx.x

    # Block index (used to determine the block's starting position in the input array)
    block_idx = cuda.blockIdx.x

    # Shared memory to store elements for this block
    shared_memory = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Compute the global index of the thread within the input array
    global_idx = block_idx * BLOCK_DIM + local_thread_idx

    # Load the thread's element into shared memory if it's within bounds
    if global_idx < size:
        shared_memory[local_thread_idx] = a[global_idx]
    else:
        # Initialize out-of-bounds threads with zero
        shared_memory[local_thread_idx] = 0.0

    # Synchronize all threads to ensure shared memory is fully populated
    cuda.syncthreads()

    # Perform reduction in shared memory using a binary tree approach
    reduction_offset = 1
    while reduction_offset < BLOCK_DIM:
        # Synchronize threads before each reduction step
        cuda.syncthreads()

        # Only threads responsible for reduction at this step participate
        if local_thread_idx % (reduction_offset * 2) == 0:
            # Sum the current element with the corresponding offset element
            shared_memory[local_thread_idx] += shared_memory[local_thread_idx + reduction_offset]

        # Double the reduction offset for the next step
        reduction_offset *= 2

    # Synchronize threads to ensure the reduction is complete
    cuda.syncthreads()

    # The first thread in the block writes the block's sum to the output array
    if local_thread_idx == 0:
        out[block_idx] = shared_memory[0]


    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")


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


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

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
    reduce_initial_value: float,
) -> None:
    """
    CUDA kernel for performing a reduction along a specified dimension.

    Args:
        out (Storage): The output storage for the result of the reduction.
        out_shape (Shape): The shape of the output tensor.
        out_strides (Strides): The strides of the output tensor.
        out_size (int): The total number of elements in the output tensor.
        a_storage (Storage): The input storage for the tensor to be reduced.
        a_shape (Shape): The shape of the input tensor.
        a_strides (Strides): The strides of the input tensor.
        reduce_dim (int): The dimension along which to perform the reduction.
        reduce_initial_value (float): The initial value for the reduction operation.
    """
    # Define the number of threads per block
    THREADS_PER_BLOCK = 1024

    # Shared memory for reduction within a block
    shared_memory = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)

    # Thread and block indices
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x

    # Size of the dimension being reduced
    reduce_dim_size = a_shape[reduce_dim]

    # Temporary array to store the index for accessing input/output tensors
    local_out_index = cuda.local.array(MAX_DIMS, numba.int32)

    # Compute the output index for this block
    to_index(block_idx, out_shape, local_out_index)

    # Compute the position in the output tensor storage
    out_storage_pos = index_to_position(local_out_index, out_strides)

    # Initialize shared memory for this thread
    if thread_idx < reduce_dim_size:
        # Update the index to include the current reduction dimension
        local_out_index[reduce_dim] = thread_idx

        # Compute the position in the input tensor storage
        input_storage_pos = index_to_position(local_out_index, a_strides)

        # Load the corresponding value from input storage to shared memory
        shared_memory[thread_idx] = a_storage[input_storage_pos]
    else:
        # If this thread is out of bounds, initialize with the reduction's identity value
        shared_memory[thread_idx] = reduce_initial_value

    # Synchronize threads to ensure shared memory is populated
    cuda.syncthreads()

    # Reduction phase: combine elements in shared memory
    reduction_offset = 1
    while reduction_offset < THREADS_PER_BLOCK:
        # Synchronize threads before each reduction step
        cuda.syncthreads()

        # Combine elements if the thread is responsible for a pair
        if thread_idx % (reduction_offset * 2) == 0:
            shared_memory[thread_idx] = fn(
                shared_memory[thread_idx],
                shared_memory[thread_idx + reduction_offset]
            )

        # Double the reduction offset
        reduction_offset *= 2

    # Synchronize threads to ensure the reduction is complete
    cuda.syncthreads()

    # Write the reduced result to the output tensor
    if thread_idx == 0:
        out[out_storage_pos] = shared_memory[0]
        
        # # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

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

    local_x = cuda.threadIdx.x
    local_y = cuda.threadIdx.y

    if idx_x < size and idx_y < size:
        shm_a[local_x, local_y] = a[index_to_position((idx_x, local_y), (size, 1))]
        shm_b[local_x, local_y] = b[index_to_position((local_x, idx_y), (size, 1))]
    else:
        shm_a[local_x, local_y] = 0.0
        shm_b[local_x, local_y] = 0.0

    cuda.syncthreads()

    total = 0.0
    for i in range(size):
        total += shm_a[local_x, i] * shm_b[i, local_y]

    if idx_x < size and idx_y < size:
        out[index_to_position((idx_x, idx_y), (size, 1))] = total


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
    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # # Batch dimension - fixed
    # batch = cuda.blockIdx.z

    # BLOCK_DIM = 32
    # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # # The final position c[i, j]
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # # The local position in the block.
    # pi = cuda.threadIdx.x
    # pj = cuda.threadIdx.y

    # # Code Plan:
    # # 1) Move across shared dimension by block dim.
    # #    a) Copy into shared memory for a matrix.
    # #    b) Copy into shared memory for b matrix
    # #    c) Compute the dot produce for position c[i, j]
    # # TODO: Implement for Task 3.4.
    # raise NotImplementedError("Need to implement for Task 3.4")

    BLOCK_DIM = 32

    # Shared memory for the tiles of A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    batch = cuda.blockIdx.z

    # Global thread indices
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Local thread indices within the block
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Extract shapes and strides
    M, K_a = a_shape[-2], a_shape[-1]
    K_b, N = b_shape[-2], b_shape[-1]

    # Assert matrix multiplication compatibility
    assert K_a == K_b

    # Stride offsets for broadcasting
    a_batch_stride = a_strides[0] if len(a_shape) > 2 else 0
    b_batch_stride = b_strides[0] if len(b_shape) > 2 else 0

    # Initialize the output value
    c_value = 0.0

    # Loop over all tiles along the K dimension
    for tile_idx in range((K_a + BLOCK_DIM - 1) // BLOCK_DIM):
        # Global memory positions for A and B
        a_row = i
        a_col = tile_idx * BLOCK_DIM + pj

        b_row = tile_idx * BLOCK_DIM + pi
        b_col = j

        # Load A into shared memory if within bounds
        if a_row < M and a_col < K_a:
            a_shared[pi, pj] = a_storage[
                index_to_position((batch, a_row, a_col), (a_batch_stride, *a_strides[1:]))
            ]
        else:
            a_shared[pi, pj] = 0.0

        # Load B into shared memory if within bounds
        if b_row < K_b and b_col < N:
            b_shared[pi, pj] = b_storage[
                index_to_position((batch, b_row, b_col), (b_batch_stride, *b_strides[1:]))
            ]
        else:
            b_shared[pi, pj] = 0.0

        # Synchronize threads to ensure shared memory is fully populated
        cuda.syncthreads()

        # Perform partial dot product for the tile
        for k in range(BLOCK_DIM):
            c_value += a_shared[pi, k] * b_shared[k, pj]

        # Synchronize threads to prevent race conditions
        cuda.syncthreads()

    # Write the computed value to global memory if within bounds
    if i < M and j < N:
        out_pos = index_to_position((batch, i, j), out_strides)
        out[out_pos] = c_value


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
