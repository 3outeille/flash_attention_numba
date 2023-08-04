import random
import os
import numpy as np
import torch
from numba import cuda, float64
import math
from pdb import set_trace

def seed_everything():
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def cpu(A, B):
    sO = A @ B
    print("---- sO ----")
    print(sO)
    tile_rowmax = torch.max(sO, dim=1).values
    tile_numerator = sO + tile_rowmax
    return tile_rowmax, tile_numerator

BLOCKSIZE = 1

@cuda.jit
def kernel(A, B, tile_rowmax, tile_numerator, tmp_O):

    sA = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float64)
    sB = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float64)
    sO = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float64)

    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    g = cuda.cg.this_grid()

    for blockId in range(cuda.gridDim.x):

        # Load a tile of A and B into shared memory
        if row < A.shape[0] and tx + blockId * BLOCKSIZE < A.shape[1]:
            sA[ty, tx] = A[row, tx + blockId * BLOCKSIZE]
        if col < B.shape[1] and ty + blockId * BLOCKSIZE < B.shape[0]:
            sB[ty, tx] = B[ty + blockId * BLOCKSIZE, col]

        cuda.syncthreads()

        # Matmul on the current tile
        for k in range(BLOCKSIZE):
            sO[ty, tx] += sA[ty, k] * sB[k, tx]

    cuda.atomic.max(tile_rowmax, row, sO[ty, tx])

    cuda.syncthreads()

    tmp_O[row, col] = sO[ty, tx]

    g.sync()

    tile_numerator[row, col] = sO[ty, tx] + tile_rowmax[col]


if __name__ == "__main__":
    seed_everything()

    N = 3

    h_A = (torch.arange(N * N, dtype=torch.float64).reshape(N, N) + 1.) / 10
    h_B = (torch.arange(N * N, dtype=torch.float64).reshape(N, N) + 1.) / 10
    h_tile_rowmax = torch.zeros(N, dtype=torch.float64)
    h_tile_numerator = torch.zeros(N * N, dtype=torch.float64).reshape(N, N)
    
    d_A = cuda.to_device(h_A)
    d_B = cuda.to_device(h_B)
    d_tile_rowmax = cuda.to_device(h_tile_rowmax)
    d_tile_numerator = cuda.to_device(h_tile_numerator)

    print("=============== CPU ====================")
    tile_rowmax_cpu, tile_numerator_cpu = cpu(h_A.clone(), h_B.clone())
    tile_rowmax_cpu = tile_rowmax_cpu.numpy()
    tile_numerator_cpu = tile_numerator_cpu.numpy()

    print("---- tile_rowmax ----")
    print(tile_rowmax_cpu)
    print("---- tile_numerator ----")
    print(tile_numerator_cpu)

    print("=============== GPU ====================")
    threadsperblock = (BLOCKSIZE, BLOCKSIZE)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}")

    h_tmp_O = torch.zeros(N * N, dtype=torch.float64).reshape(N, N)
    d_tmp_O = cuda.to_device(h_tmp_O)

    kernel[blockspergrid, threadsperblock](
        d_A,
        d_B,
        d_tile_rowmax,
        d_tile_numerator,
        d_tmp_O,
    )
    cuda.synchronize()

    h_tmp_O = d_tmp_O.copy_to_host()
    h_tile_rowmax = d_tile_rowmax.copy_to_host()
    h_tile_numerator = d_tile_numerator.copy_to_host()
    
    print("---- tmp_O ----")
    print(h_tmp_O)
    print("---- tile_rowmax ----")
    print(h_tile_rowmax)
    print("---- tile_numerator ----")
    print(h_tile_numerator)

    assert np.allclose(h_tile_rowmax, tile_rowmax_cpu)
    assert np.allclose(h_tile_numerator, tile_numerator_cpu)