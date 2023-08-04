import numpy as np
from numba import cuda, float32
import math

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of BLOCKSIZExBLOCKSIZE elements.
# BLOCKSIZE should not be larger than 32 in this example (because max threads per block is 1024, and 32x32=1024)
BLOCKSIZE = 2

@cuda.jit
def kernel_matmul(A, B, C):
    sA = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float32)
    sB = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float32)

    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    tmp = float32(0.)

    # print("(row, col): ", row, col, "| (tx, ty): ", tx, ty)
    
    # Loop until all tiles of a given direction are processed 
    # (in order to have a complete matrix multiplication along this direction)
    for blockId in range(cuda.gridDim.x):

        # Load a tile of A and B into shared memory
        if row < A.shape[0] and tx + blockId * BLOCKSIZE < A.shape[1]:
            sA[ty, tx] = A[row, tx + blockId * BLOCKSIZE]
        if col < B.shape[1] and ty + blockId * BLOCKSIZE < B.shape[0]:
            sB[ty, tx] = B[ty + blockId * BLOCKSIZE, col]
                    
        cuda.syncthreads()

        for k in range(BLOCKSIZE):
            tmp += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp


if __name__ == "__main__":
    # This version also works for non-square matrices
    N = 16
    size = int(np.sqrt(N))

    h_a = np.arange(N, dtype=np.float32).reshape([size, size]) + 1 
    h_b = np.arange(N, dtype=np.float32).reshape([size, size]) + 1
    h_actual = np.zeros([size, size], dtype=np.float32)
    print("--- h_a ---")
    print(h_a)

    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
    d_actual = cuda.to_device(h_actual)

    threadsperblock = (BLOCKSIZE, BLOCKSIZE)
    blockspergrid_x = math.ceil(h_actual.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(h_actual.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}")

    kernel_matmul[blockspergrid, threadsperblock](d_a, d_b, d_actual)
    
    expected = h_a @ h_b
    h_actual = d_actual.copy_to_host()
    print("--- expected ---")
    print(expected)
    print("--- actual ---")
    print(h_actual)
    print("--- expected - actual ---")
    print(expected - h_actual)
    assert np.allclose(expected, h_actual)