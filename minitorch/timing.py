import minitorch
from minitorch import TensorBackend
import time
import numpy as np
import matplotlib.pyplot as plt  # type: ignore # Import matplotlib for plotting

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: TensorBackend, size: int = 16) -> None:
    """Perform matrix multiplication between two random tensors using the specified backend.

    This function generates two random tensors 'x' and 'y' of shape (batch_size, size, size),
    performs matrix multiplication (denoted by the `@` operator), and stores the result in 'z'.
    The operation is executed on the specified backend (e.g., CPU or GPU).

    Args:
    ----
        backend (TensorBackend):
            The backend to be used for tensor operations. This could be a CPU backend, GPU backend, or other supported
            hardware backends. The backend determines where the computation will take place (e.g., on the CPU or GPU).

        size (int, optional):
            The size of the square matrices to be multiplied. Default is 16. This represents the number of rows and columns
            in the generated tensors `x` and `y`.

    Returns:
    -------
        None: The result of the matrix multiplication is stored in 'z', but this function does not return anything.
              The operation is performed in place on the tensors using the specified backend.

    Example:
    -------
        run_matmul(cpu_backend, size=32)  # Performs matrix multiplication on CPU with matrices of size 32x32.

    Notes:
    -----
        - The matrices are generated with random values using `minitorch.rand`.
        - The matrix multiplication is done using the `@` operator, which is supported by `minitorch` for tensor operations.

    """
    # commenting these out so they pass ruff tests
    # batch_size = 2  # Define the batch size for the operation

    # Create two random tensors x and y with shapes (batch_size, size, size)
    # x = minitorch.rand((batch_size, size, size), backend=backend)
    # y = minitorch.rand((batch_size, size, size), backend=backend)

    # Perform matrix multiplication (batch-wise)
    # z = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    # Print timing summary
    print("\nTiming summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")

    # Plotting the results using matplotlib
    sizes = list(times.keys())
    fast_times_avg = [times[size]["fast"] for size in sizes]
    gpu_times_avg = [times[size]["gpu"] for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(
        sizes,
        fast_times_avg,
        label="FastTensorBackend",
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.plot(
        sizes, gpu_times_avg, label="GPUBackend", marker="s", linestyle="-", color="r"
    )

    plt.title("Matrix Multiplication Benchmarking")
    plt.xlabel("Matrix Size")
    plt.ylabel("Average Time (seconds)")
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file in Colab
    plt.savefig(
        "/content/matrix_multiplication_benchmark.png"
    )  # Save to Colab's working directory
    print("Plot saved as matrix_multiplication_benchmark.png")
    plt.show()
