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


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Compile a Python function into a CUDA kernel optimized for device execution.

    This function uses a JIT (Just-In-Time) compiler to transform a regular Python
    function into a CUDA-compatible kernel that can be executed directly on a GPU device.
    It enables the use of GPU-specific optimizations by setting `device=True`.

    Args:
    ----
    fn (Fn): The Python function to be compiled into a CUDA kernel. It should be compatible
             with device execution, meaning it must be capable of handling GPU-based data
             and operations.
    **kwargs (Any): Additional keyword arguments to customize the JIT compilation process.

    Returns:
    -------
    Fn: A CUDA-compatible kernel function optimized for execution on a GPU device.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable[..., Any], **kwargs: Any) -> FakeCUDAKernel:
    """Compile a Python function into a CUDA kernel for GPU execution.

    This function uses a JIT (Just-In-Time) compiler to transform a regular Python
    function into a CUDA-compatible kernel that can be executed on a GPU. Additional
    keyword arguments can be passed to customize the compilation behavior.

    Args:
    ----
    fn (Callable[..., Any]): The Python function to be compiled into a CUDA kernel.
    **kwargs (Any): Additional keyword arguments to be passed to the underlying
                    JIT compiler for customization.

    Returns:
    -------
    FakeCUDAKernel: A CUDA-compatible kernel ready to be executed on a GPU.

    """
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
        """Applies a binary operation element-wise on two tensors using broadcasting.

        This function performs an element-wise operation on two tensors (`a` and `b`) using
        a specified binary function `fn`. The tensors are broadcasted to a common shape if
        their shapes differ, and the result is stored in a new tensor. The computation
        is performed in parallel on the GPU using CUDA for efficiency.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that takes two float arguments
                and returns a single float. This function is applied element-wise to the input tensors.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that performs the element-wise operation.
                It takes two tensors (`Tensor`, `Tensor`) as arguments and returns a new tensor
                containing the result of applying the binary function element-wise.

        Example:
        -------
        - result = zip(lambda x, y: x + y)(tensor_a, tensor_b)  # Performs element-wise addition.

        Notes:
        -----
        - The input tensors are broadcasted to a common shape before the operation if needed.
        - The function leverages GPU parallelism using CUDA to speed up the element-wise computation.
        - The result tensor will have a shape that is the result of broadcasting the input shapes.

        """
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
        """Reduces a tensor along a specified dimension using a binary operation.

        This function performs a reduction operation (such as sum, min, max, etc.) on a tensor
        along a given dimension. It applies the binary function `fn` iteratively to combine
        elements of the tensor along the specified dimension, starting with the value `start`.
        The operation is performed in parallel using CUDA for efficient execution on the GPU.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that takes two float arguments
                and returns a single float. This function is applied repeatedly to reduce the tensor
                along the specified dimension.
            start (float, optional): The initial value to start the reduction with. Default is 0.0.
                This value is used as the starting point for the reduction operation.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A function that performs the reduction operation.
                It takes two arguments: the input tensor (`Tensor`) and the dimension (`int`) along
                which the reduction is to be performed. The result is a new tensor with the reduced values.

        Example:
        -------
        - result = reduce(lambda x, y: x + y)(tensor_a, dim=0)  # Perform sum along dimension 0.

        Notes:
        -----
        - The operation is performed in parallel on the GPU using CUDA.
        - The size of the reduction operation is determined by the number of blocks and threads,
        based on the input tensor size and the specified dimension.

        """
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
        """Perform matrix multiplication between two tensors 'a' and 'b'. If either of the
        tensors is 2D, it is reshaped to 3D to allow for broadcasting and batch-wise
        multiplication. The function ensures that the last dimension of 'a' matches the
        second-to-last dimension of 'b'.

        Args:
        ----
        a (Tensor): The first input tensor. It can have any number of dimensions,
            but the last two dimensions must be compatible for matrix multiplication.
        b (Tensor): The second input tensor. It can also have any number of dimensions,
            and its last two dimensions must match the corresponding dimensions of 'a'
            for matrix multiplication.

        Returns:
        -------
        - Tensor: A tensor containing the result of the matrix multiplication. The output
            tensor will have dimensions based on the broadcasting of the input shapes,
            and its last two dimensions correspond to the result of multiplying the last
            dimension of 'a' with the second-to-last dimension of 'b'.

        Raises:
        ------
        - AssertionError: If the last dimension of 'a' does not match the second-to-last
            dimension of 'b', matrix multiplication is not possible.

        Notes:
        -----
        - If either input tensor is 2D, it will be reshaped to 3D by adding a leading
            dimension to support batch-wise multiplication.
        - This function uses a CUDA kernel (`tensor_matrix_multiply`) for efficient matrix
            multiplication on the GPU.
        - The result is broadcasted to handle different batch sizes and shapes.

        Example:
        -------
        - result = Tensor.matrix_multiply(tensor_a, tensor_b)

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

        # Perform broadcasting to align the shapes of a and b for matrix multiplication
        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert (
            a.shape[-1] == b.shape[-2]
        )  # Ensure valid matrix multiplication dimensions
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra columns
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        # Launch the CUDA kernel for matrix multiplication
        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3D reshaping if it was added
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


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """A practice sum kernel to prepare for a reduction operation.

    Given an input array `a` of length `size`, the goal is to sum up each block
    of `BLOCK_DIM` elements into a single cell in the `out` array. The length
    of the `out` array should be `size // BLOCK_DIM`, since each block of `BLOCK_DIM`
    values in `a` will correspond to a single output value in `out`.

    The input array `a` is conceptually divided into chunks (or "blocks") of `BLOCK_DIM` elements:

        Input:  [a_1, a_2, ..., a_{size}]
        Output: [sum(a_1 ... a_{BLOCK_DIM}), sum(a_{BLOCK_DIM+1} ... a_{2*BLOCK_DIM}), ...]

    Note:
    ----
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
    """Function performs a parallel sum operation on the input tensor 'a' using CUDA.

    The input tensor 'a' is summed in parallel across all elements, and the result is stored
    in a two-element tensor, where the sum is stored in the first element.

    Args:
    ----
    a (Tensor): The input tensor whose elements are to be summed. The tensor is expected
                  to have a shape that can be divided by `THREADS_PER_BLOCK`.

    Returns:
    -------
    - TensorData: A tensor of size 2, where the first element contains the sum of all elements
                  from the input tensor 'a'. The second element is reserved for any additional
                  data if needed, and it's initialized to 0.0.

    Notes:
    -----
    - The computation is performed using GPU via CUDA and `jit_sum_practice` kernel.
    - The size of the input tensor 'a' is divided into blocks and threads for parallel processing.

    """
    (size,) = a.shape  # Extract the size of the tensor
    threadsperblock = (
        THREADS_PER_BLOCK  # Define the number of threads per block for CUDA
    )
    blockspergrid = (
        size // THREADS_PER_BLOCK
    ) + 1  # Calculate the number of blocks required
    out = TensorData(
        [0.0 for i in range(2)], (2,)
    )  # Initialize the output tensor with two elements
    out.to_cuda_()  # Move the output tensor to CUDA memory
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0],
        a._tensor._storage,
        size,  # Launch the CUDA kernel to perform the sum
    )
    return out  # Return the result tensor containing the sum


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
        """The CUDA kernel function for performing the tensor reduction along a specified dimension.

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
    """Practice square MM kernel to prepare for matmul.

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
    # Set the block dimension to 32, which will be used for shared memory size and grid/block layout
    BLOCK_DIM = 32

    # Allocate shared memory arrays for matrices 'a' and 'b' with dimensions BLOCK_DIM x BLOCK_DIM
    # Shared memory is faster for threads within the same block to access
    shm_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shm_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Calculate the global x and y indices for the current thread in the grid
    # These represent the position of the thread within the entire matrix
    idx_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idx_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # If the indices are outside the bounds of the matrix size, return early to avoid accessing invalid memory
    if idx_x >= size or idx_y >= size:
        return

    # Convert the global indices (idx_x, idx_y) into a linear position in the matrix
    # The second argument (size, 1) is used to interpret the global index as a 1D array position
    pos = index_to_position((idx_x, idx_y), (size, 1))

    # Load the matrix elements 'a' and 'b' from global memory into shared memory
    # This is done to reduce global memory accesses, since shared memory is much faster
    shm_a[idx_x][idx_y] = a[pos]
    shm_b[idx_x][idx_y] = b[pos]

    # Synchronize threads within the block to ensure all threads have completed loading data into shared memory
    cuda.syncthreads()

    # Initialize the variable that will hold the result of the matrix multiplication
    total = 0.0

    # Perform the matrix multiplication in a loop over the third dimension (i.e., size of the matrix)
    # Each thread computes one element of the resulting matrix
    for i in range(size):
        total += shm_a[idx_x][i] * shm_b[i][idx_y]

    # Store the result of the multiplication in the output matrix at the correct position
    out[pos] = total


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform matrix multiplication on two input tensors 'a' and 'b'. The function
    multiplies two 2D matrices (a and b) and stores the result in a tensor of shape
    (size, size), where 'size' is the number of rows/columns in the input tensors.

    The function uses a CUDA kernel (`jit_mm_practice`) to perform the matrix
    multiplication in parallel on the GPU.

    Args:
    ----
        a (Tensor):
            The first input tensor, a 2D matrix with shape (size, size). This tensor represents the
            left operand in the matrix multiplication.

        b (Tensor):
            The second input tensor, a 2D matrix with shape (size, size). This tensor represents the
            right operand in the matrix multiplication.

    Returns:
    -------
    - TensorData: A new tensor containing the result of the matrix multiplication.
      The output tensor has shape (size, size).

    Example:
    -------
    - result = mm_practice(tensor_a, tensor_b)

    Notes:
    -----
    - The input tensors must have matching dimensions for matrix multiplication (i.e.,
      both should be 2D with shape (size, size)).
    - The function leverages GPU parallelism using CUDA to accelerate the computation.

    """
    # Get the size of the input tensor 'a'. The second dimension is ignored since both
    # 'a' and 'b' are assumed to have shape (size, size).
    (size, _) = a.shape

    # Define the number of threads per block (2D grid)
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)

    # Number of blocks per grid (set to 1 for simplicity, assuming the operation fits in a single block)
    blockspergrid = 1

    # Create an output tensor of size (size, size) initialized to zeros
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()

    # Launch the CUDA kernel to perform matrix multiplication
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )

    return out


def _tensor_matrix_multiply(
    out: Storage,  # Output storage where the result of the multiplication will be stored
    out_shape: Shape,  # Shape of the output tensor (dimensions)
    out_strides: Strides,  # Strides of the output tensor, used for memory access
    out_size: int,  # Total size of the output tensor
    a_storage: Storage,  # Storage for input tensor 'a'
    a_shape: Shape,  # Shape of the input tensor 'a'
    a_strides: Strides,  # Strides of the input tensor 'a'
    b_storage: Storage,  # Storage for input tensor 'b'
    b_shape: Shape,  # Shape of the input tensor 'b'
    b_strides: Strides,  # Strides of the input tensor 'b'
) -> None:
    """Performs matrix multiplication on two tensors 'a' and 'b' and stores the result in 'out'.
    This function is designed for efficient execution on the GPU using CUDA.

    The multiplication is done in blocks, where shared memory is used to reduce global memory
    accesses. Each thread computes a portion of the output matrix by performing a dot product
    between a row of 'a' and a column of 'b'.

    Args:
    ----
        out (Storage):
            The output storage where the result of the matrix multiplication will be stored. This tensor
            must have a shape that matches the result of multiplying the two input tensors 'a' and 'b'.

        out_shape (Shape):
            The shape of the output tensor, typically a tuple of the form (batch_size, M, N), where M and N
            represent the dimensions of the output matrix for each batch.

        out_strides (Strides):
            The memory strides for the output tensor. Strides define the step size in memory between consecutive
            elements along each dimension of the tensor.

        out_size (int):
            The total number of elements in the output tensor. This is used to determine the size of the tensor
            and the range of threads needed to compute the result.

        a_storage (Storage):
            The storage for the input tensor 'a'. This tensor holds the data for the left operand in the
            matrix multiplication.

        a_shape (Shape):
            The shape of the input tensor 'a', which should have the form (batch_size, M, K) for matrix
            multiplication with 'b' of shape (batch_size, K, N).

        a_strides (Strides):
            The memory strides for the input tensor 'a'. These define the memory layout and how to navigate
            through the elements of 'a'.

        b_storage (Storage):
            The storage for the input tensor 'b'. This tensor holds the data for the right operand in the
            matrix multiplication.

        b_shape (Shape):
            The shape of the input tensor 'b', which should have the form (batch_size, K, N) for matrix
            multiplication with 'a' of shape (batch_size, M, K).

        b_strides (Strides):
            The memory strides for the input tensor 'b', which define the memory layout and how to access
            the elements of 'b'.

    Returns:
    -------
    - None: The result is directly stored in the 'out' storage.

    Description:
    - The function uses CUDA block and thread indices to divide the computation of the output matrix
      into smaller blocks of data, which are processed in parallel by the GPU.
    - Shared memory arrays (`a_shared` and `b_shared`) are used within each block to hold submatrices
      from 'a' and 'b'. This reduces memory access latency and improves performance.
    - The matrix multiplication is carried out in multiple stages (in terms of blocks), where each
      thread loads its part of the data, performs the dot product for the corresponding element in
      the output matrix, and then stores the result.
    - Batch processing is supported, where each block can handle a different batch in the input tensors.

    """
    # Compute the batch strides for 'a' and 'b'. These are used to navigate through different batches.
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Get the batch index for the current thread. It is used for parallel processing of different batches.
    batch = cuda.blockIdx.z

    # Define the block size (32x32) for the shared memory.
    BLOCK_DIM = 32

    # Allocate shared memory arrays for blocks of 'a' and 'b'. This improves performance by reducing global memory access.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Calculate the global row (i) and column (j) indices for the current thread.
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Get the thread-specific indices within the block
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Extract the dimensions for the output tensor and the last dimension of 'a' (K).
    M, N, K = out_shape[1], out_shape[2], a_shape[-1]

    # Initialize the accumulator to 0, which will hold the result of the dot product for this thread.
    acc = 0.0

    # Loop through the matrix 'a' and 'b' in BLOCK_DIM-sized chunks along the K dimension.
    for start in range(
        0, K, BLOCK_DIM
    ):  # 'start' marks the starting index for the block
        # Calculate the k-th element of 'a' that corresponds to the current block and thread.
        a_k = start + pj

        # Load data from 'a_storage' into the shared memory 'a_shared' for the current block.
        if i < M and a_k < K:
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[1] + a_k * a_strides[2]
            ]

        # Calculate the k-th element of 'b' that corresponds to the current block and thread.
        b_k = start + pi

        # Load data from 'b_storage' into the shared memory 'b_shared' for the current block.
        if b_k < K and j < N:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + b_k * b_strides[1] + j * b_strides[2]
            ]

        # Synchronize threads within the block to ensure all data is loaded into shared memory
        # before proceeding with the computation.
        cuda.syncthreads()

        # Perform the dot product calculation for the block. Each thread computes a portion of the sum.
        for k in range(BLOCK_DIM):
            # Ensure that 'k' is within the bounds of the matrix dimensions
            if start + k < K:
                acc += a_shared[pi, k] * b_shared[k, pj]

    # If the current thread is within the bounds of the output tensor, store the result.
    # The result is stored in the correct position based on the global indices (i, j).
    if i < M and j < N:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)  # type: ignore
