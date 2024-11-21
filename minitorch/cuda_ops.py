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

    # def _reduce(
    #     out: Storage,
    #     out_shape: Shape,
    #     out_strides: Strides,
    #     out_size: int,
    #     a_storage: Storage,
    #     a_shape: Shape,
    #     a_strides: Strides,
    #     reduce_dim: int,
    #     reduce_value: float,
    # ) -> None:
    #     BLOCK_DIM = 1024
    #     # cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    #     # out_index = cuda.local.array(MAX_DIMS, numba.int32)
    #     # out_pos = cuda.blockIdx.x
    #     # pos = cuda.threadIdx.x
    #     reduce_size = a_shape[reduce_dim]
    #     local_idx = cuda.threadIdx.x
    #     block_idx = cuda.blockIdx.x
    #     shared_block = cuda.shared.array(BLOCK_DIM, numba.float64)
    #     offset = 1
    #     out_index = cuda.local.array(MAX_DIMS, numba.int32)
    #     to_index(block_idx, out_shape, out_index)
    #     out_position = index_to_position(out_index, out_strides)
    #     if local_idx < reduce_size:
    #         out_index[reduce_dim] = local_idx
    #         shared_block[local_idx] = a_storage[index_to_position(out_index, a_strides)]
    #     else:
    #         shared_block[local_idx] = reduce_value
    #     while offset < BLOCK_DIM:
    #         cuda.syncthreads()
    #         if local_idx % (offset * 2) == 0:
    #             shared_block[local_idx] = fn(
    #                 shared_block[local_idx], shared_block[local_idx + offset]
    #             )
    #         offset *= 2
    #     cuda.syncthreads()
    #     if local_idx == 0:
    #         out[out_position] = shared_block[local_idx]

    # return jit(_reduce)  # type: ignore
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
    """Performs a reduction operation along a specific dimension of the input tensor.

    This function reduces the elements of the `a_storage` tensor along the `reduce_dim`
    dimension using a specified reduction function. The result of the reduction is
    stored in the `out` tensor.

    Args:
    ----
        out (Storage): The storage object where the output of the reduction is stored.
        out_shape (Shape): The shape of the output tensor.
        out_strides (Strides): The strides of the output tensor.
        out_size (int): The total number of elements in the output tensor.
        a_storage (Storage): The storage object for the input tensor to be reduced.
        a_shape (Shape): The shape of the input tensor.
        a_strides (Strides): The strides of the input tensor.
        reduce_dim (int): The dimension along which the reduction is performed.
        reduce_value (float): The initial value used for reduction (e.g., 0 for sum).

    Note:
    -----
    The reduction operation is done using shared memory to enhance parallel computation.
    Each block in CUDA reduces its corresponding segment of the input tensor and writes
    the result to the output tensor.

    """

    # The size of each block of threads (number of threads in a block).
    BLOCK_DIM = 1024

    # Compute the number of elements in the dimension being reduced.
    # This determines how many elements will be summed/reduced in each block.
    reduce_size = a_shape[reduce_dim]

    # Calculate the local thread index (within a block) and the block index.
    # `local_idx` refers to the index of the current thread within a block.
    # `block_idx` is the index of the block within the grid of blocks.
    local_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x

    # Shared memory allocation for the current block. This shared memory allows
    # threads within the same block to communicate and collaborate on data processing.
    shared_block = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Allocate a local array to store the output index temporarily.
    # This array is used to track the indices within the output tensor.
    out_index = cuda.local.array(MAX_DIMS, numba.int32)

    # Convert the current block index (`block_idx`) to a multi-dimensional index
    # that corresponds to the `out_shape`. This is useful for navigating the
    # higher-dimensional tensor space.
    to_index(block_idx, out_shape, out_index)

    # Compute the output position in the flattened output storage using the
    # `out_strides`. This gives the linear index where the reduction result
    # should be stored for the current block.
    out_position = index_to_position(out_index, out_strides)

    # Populate the shared memory with elements from the input tensor.
    # Each thread is responsible for a specific element based on its local index.
    # If the thread's local index exceeds the dimension size being reduced, it
    # initializes that part of shared memory with the `reduce_value` (default value).
    if local_idx < reduce_size:
        # Adjust the out_index to point to the specific slice being reduced.
        out_index[reduce_dim] = local_idx
        # Compute the position in the input tensor's storage based on the updated index.
        input_position = index_to_position(out_index, a_strides)
        shared_block[local_idx] = a_storage[input_position]
    else:
        # If the local index exceeds the size of the reduce dimension, fill with reduce_value.
        shared_block[local_idx] = reduce_value

    # Synchronize all threads in the block to ensure shared memory is populated
    # correctly before proceeding with the reduction.
    cuda.syncthreads()

    # Perform the reduction using a tree-based approach. This approach gradually
    # combines pairs of elements in shared memory until only one value remains.
    offset = 1
    while offset < BLOCK_DIM:
        # Synchronize all threads before performing each reduction step to ensure
        # that all threads have access to the correct values in shared memory.
        cuda.syncthreads()

        # Only threads whose local index is a multiple of `2 * offset` participate
        # in this step. Each such thread combines its current value with the value
        # located `offset` positions away.
        if local_idx % (offset * 2) == 0:
            shared_block[local_idx] = fn(
                shared_block[local_idx], shared_block[local_idx + offset]
            )

        # Double the offset to move to the next level of reduction.
        offset *= 2

    # Synchronize once more to ensure all reductions are completed before writing
    # the final result.
    cuda.syncthreads()

    # After the reduction is complete, the result is stored in `shared_block[0]`.
    # The first thread in each block writes this value to the correct position
    # in the output storage.
    if local_idx == 0:
        out[out_position] = shared_block[0]
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
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
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
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    y = cuda.threadIdx.y
    x = cuda.threadIdx.x
    block_x = cuda.blockIdx.x * BLOCK_DIM
    block_y = cuda.blockIdx.y * BLOCK_DIM
    z = cuda.blockIdx.z
    temp = 0
    for block_index in range((a_shape[-1] + (BLOCK_DIM - 1)) // BLOCK_DIM):
        block_mid = block_index * BLOCK_DIM
        if (block_mid + x) < a_shape[-1] and (block_y + y) < a_shape[-2]:
            shared_a[y, x] = a_storage[
                z * a_batch_stride
                + (block_mid + x) * a_strides[-1]
                + (block_y + y) * a_strides[-2]
            ]
        else:
            shared_a[y, x] = 0
        if (block_x + x) < b_shape[-1] and (block_mid + y) < b_shape[-2]:
            shared_b[y, x] = b_storage[
                z * b_batch_stride
                + (block_x + x) * b_strides[-1]
                + (block_mid + y) * b_strides[-2]
            ]
        else:
            shared_b[y, x] = 0
        cuda.syncthreads()
        for val in range(BLOCK_DIM):
            temp += shared_a[y, val] * shared_b[val, x]
    if (block_y + y) < out_shape[-2] and (block_x + x) < out_shape[-1]:
        out[
            z * out_strides[0]
            + (block_y + y) * out_strides[-2]
            + (block_x + x) * out_strides[-1]
        ] = temp

tensor_matrix_multiply = jit(_tensor_matrix_multiply)
