from numba import cuda
import numpy as np

@cuda.jit
def kernel_dtype_working():
    f32_arr = cuda.shared.array(0, dtype=np.float64)
    f32_arr[0] = 3.14
    print(f32_arr[0])

@cuda.jit
def kernel_dtype_notworking(dtype):
    f32_arr = cuda.shared.array(0, dtype=dtype)
    f32_arr[0] = 3.14
    print(f32_arr[0])

if __name__ == "__main__":
    kernel_dtype_working[1, 1, 0, 8]()
    cuda.synchronize()
    kernel_dtype_notworking[1, 1, 0, 8](np.float64)
    cuda.synchronize()
