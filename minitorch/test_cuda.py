from numba import cuda


def test_cuda() -> None:
    """Check if CUDA is available and print the current CUDA device information.

    This function checks whether a CUDA-capable GPU is available and prints the
    name of the current device if available. If CUDA is not available, it prints
    a corresponding message.

    Returns
    -------
    None

    """
    if cuda.is_available():
        print("CUDA is available!")
        device = cuda.get_current_device()
        print(f"Device name: {device.name}")
    else:
        print("CUDA is not available.")


test_cuda()
