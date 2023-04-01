import numpy as np
from numba import cuda, float64
import math
import torch

def ref_attention(Q, K, V):
    # B, C, SEQ_LEN, D_HEAD = (3, 2, 2, 1024)
    S = torch.matmul(Q, K.transpose(2, 3))
    # (3, 2, 2, 2)
    # Compute softmax for each row of S
    # P = torch.softmax(S, dim=2)
    # O =  torch.matmul(P, V)
    # return O
    return S

@cuda.jit
def flash_attention(Q, K, V, O, smem_offset, S_debug):
    tid = cuda.threadIdx.x

    sQ = cuda.shared.array(0, dtype=float64)[:smem_offset] # 32 * 8 = 256
    sK = cuda.shared.array(0, dtype=float64)[smem_offset:2*smem_offset]
    sO = cuda.shared.array(0, dtype=float64)[2*smem_offset:3*smem_offset]
    sS = cuda.shared.array(0, dtype=float64)[3*smem_offset:] # 32 * 32 * 8 = 8192

    # 3rd dim is always at 0 because from now on, we treat Q as (B, C, SEQ_LEN * D_HEAD)
    # Load Q into shared memory
    sQ[tid] = Q[cuda.blockIdx.y, cuda.blockIdx.z, 0, tid + cuda.blockIdx.x * cuda.blockDim.x]
    # sK[tid] = K[cuda.blockIdx.y, cuda.blockIdx.z, 0, tid + cuda.blockIdx.x * cuda.blockDim.x]

    cuda.syncthreads()

    # tmp = float64(0)
    # tmp += sQ[tid] * sK[tid]
    #TODO: Perform Matmul in 4D and stored in S_debug 
    if cuda.blockIdx.x == 1 and cuda.blockIdx.y == 0 and cuda.blockIdx.z == 0:
        print(tid, sQ[tid])
    
    cuda.syncthreads()

    # O[cuda.blockIdx.y, cuda.blockIdx.z, 0, tid + cuda.blockIdx.x * cuda.blockDim.x] = sS[tid]


if __name__ == "__main__":
    B, C, SEQ_LEN, D_HEAD = (3, 2, 2, 32)
    Q = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    K = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    V = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    # TODO: init O with zeros inside the kernel
    O = torch.zeros((B,  C, SEQ_LEN, D_HEAD), dtype=torch.float64)

    ref = ref_attention(Q.clone(), K.clone(), V.clone())

    d_Q = cuda.to_device(Q)
    d_K = cuda.to_device(K)
    d_V = cuda.to_device(V)
    d_O = cuda.to_device(O)

    # TODO: Find best number of threads and blocks given compute capability given MAX Shareed Memory size for my gpu 
    stream = cuda.default_stream()
    threadsperblock = (32, 1, 1)
    blockspergrid = (math.ceil(Q.shape[2] * Q.shape[3] / threadsperblock[0]), B, C)
    shared_memory_size = (threadsperblock[0] * np.dtype(np.float64).itemsize) * 3 + (SEQ_LEN**2 * np.dtype(np.float64).itemsize)
    smem_offset = threadsperblock[0] * np.dtype(np.float64).itemsize
    print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}, shared_memory_size: {shared_memory_size}, smem_offset: {smem_offset}")

    d_S_debug = cuda.to_device(torch.zeros((B, C, SEQ_LEN, SEQ_LEN), dtype=torch.float64))

    flash_attention[blockspergrid, threadsperblock, stream, shared_memory_size](
        d_Q,
        d_K,
        d_V,
        d_O,
        smem_offset,
        # DEBUG
        d_S_debug
    )
    
    # pred = torch.from_numpy(d_O.copy_to_host())
    # assert torch.allclose(ref, pred)

    S_debug = torch.from_numpy(d_S_debug.copy_to_host())

    assert torch.allclose(ref, S_debug)