from numba import cuda
import numpy as np

@cuda.jit
def f_with_view():
    f32_arr = cuda.shared.array(0, dtype=np.float32)
    i32_arr = cuda.shared.array(0, dtype=np.int32)[1:]
    f32_arr[0] = 3.14
    i32_arr[0] = 1
    print(f32_arr[0])
    print(i32_arr[0])


f_with_view[1, 1, 0, 8]()
cuda.synchronize()