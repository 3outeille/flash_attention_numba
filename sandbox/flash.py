import numpy as np
from numba import cuda, float64, int64
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
def flash_attention(Q, K, V, O, S_debug):
    SEQ_LEN = Q.shape[2]
    D_HEAD = Q.shape[3]
    BLOCK_SIZE = cuda.blockDim.x
    tid = cuda.threadIdx.x

    sQ = cuda.shared.array(0, dtype=np.float64)[:D_HEAD]
    sK = cuda.shared.array(0, dtype=np.float64)[D_HEAD:2*D_HEAD]

    if tid >= D_HEAD:
        return

    Q_offset = int64(0)

    for i in range(SEQ_LEN):

        for l in range(D_HEAD // BLOCK_SIZE):
            # 3rd dim is always at 0 because from now on, we treat Q as (B, C, SEQ_LEN * D_HEAD)
            sQ[tid + l * BLOCK_SIZE] = Q[cuda.blockIdx.y, cuda.blockIdx.z, 0, tid + (i + l) * BLOCK_SIZE + Q_offset]

        cuda.syncthreads()
        
        K_offset = int64(0)

        for j in range(SEQ_LEN):
            
            for l in range(D_HEAD // BLOCK_SIZE):
                sK[tid + l * BLOCK_SIZE] = K[cuda.blockIdx.y, cuda.blockIdx.z, 0, tid + (j + l) * BLOCK_SIZE + K_offset]

            cuda.syncthreads()

            tmp = float64(0)

            for k in range(D_HEAD):
                tmp += sQ[k] * sK[k]

            S_debug[cuda.blockIdx.y, cuda.blockIdx.z, i, j] = tmp
            
            K_offset += (D_HEAD // BLOCK_SIZE - 1)*BLOCK_SIZE

            # if cuda.blockIdx.x == 0 and cuda.blockIdx.y == 0 and cuda.blockIdx.z == 0:
            #     if tid == 0:
            #         for i in range(D_HEAD):
            #             print(i, sQ[i], sK[i])

        Q_offset += (D_HEAD // BLOCK_SIZE - 1)*BLOCK_SIZE


if __name__ == "__main__":
    B, C, SEQ_LEN, D_HEAD = (3, 1, 2, 32) #128
    Q = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    K = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    V = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    # TODO: init O with zeros inside the kernel
    O = torch.zeros((B, C, SEQ_LEN, D_HEAD), dtype=torch.float64)

    ref = ref_attention(Q.clone(), K.clone(), V.clone())

    d_Q = cuda.to_device(Q)
    d_K = cuda.to_device(K)
    d_V = cuda.to_device(V)
    d_O = cuda.to_device(O)

    # TODO: Find best number of threads and blocks given compute capability given MAX Shareed Memory size for my gpu 
    stream = cuda.default_stream()
    threadsperblock = (32, 1, 1)
    blockspergrid = (math.ceil(Q.shape[2] * Q.shape[3] / threadsperblock[0]), B, C)
    shared_memory_size = (D_HEAD * np.dtype(np.float64).itemsize) * 2# + (SEQ_LEN**2 * np.dtype(np.float64).itemsize)
    print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}, shared_memory_size: {shared_memory_size}")

    # ====== DEBUG ======
    d_S_debug = cuda.to_device(torch.zeros((B, C, SEQ_LEN, SEQ_LEN), dtype=torch.float64))

    flash_attention[blockspergrid, threadsperblock, stream, shared_memory_size](
        d_Q,
        d_K,
        d_V,
        d_O,
        # DEBUG
        d_S_debug
    )
    
    cuda.synchronize()

    S_debug = torch.from_numpy(d_S_debug.copy_to_host())
    print(ref)
    print("=====")
    print(S_debug)

    assert torch.allclose(ref, S_debug)