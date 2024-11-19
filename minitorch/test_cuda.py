from numba import cuda

def test_cuda():
    if cuda.is_available():
        print("CUDA is available!")
        device = cuda.get_current_device()
        print(f"Device name: {device.name}")
    else:
        print("CUDA is not available.")

test_cuda()