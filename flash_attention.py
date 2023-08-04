import numpy as np
from numba import cuda, float64
import math
import torch
import time

def get_torch_dtype(str):
    if str == "float32":
        return torch.float32
    elif str == "float64":
        return torch.float64
    else:
        raise ValueError(f"Unknown dtype: {str}")
    
def torch2numpy_dtype(dtype):
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.float64:
        return np.float64
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def ref_attention(Q, K, V):
    #TODO: divide by sqrt(d_k)
    S = torch.matmul(Q, K.transpose(-2, -1))
    P = torch.softmax(S, dim=-1)
    O =  torch.matmul(P, V)
    return O

@cuda.jit
def _flash_attention_forward_kernel(Q, K, V, O):

    # TODO: Handle multiple blocks
    # TODO: Handle K transpose loading
    # TODO: divide by sqrt(dk)

    SEQ_LEN = Q.shape[2]
    D_HEAD = Q.shape[3]
    BLOCK_SIZE = cuda.blockDim.x
    tid = cuda.threadIdx.x
    
    sQ = cuda.shared.array(0, dtype=O.dtype)[:D_HEAD]
    sK = cuda.shared.array(0, dtype=O.dtype)[D_HEAD:2*D_HEAD]
    sO = cuda.shared.array(0, dtype=O.dtype)[2*D_HEAD:3*D_HEAD]
    sS = cuda.shared.array(0, dtype=O.dtype)[3*D_HEAD:3*D_HEAD + SEQ_LEN**2]

    if tid >= D_HEAD:
        return

    prev_rowmax = float64(-np.inf)
    tile_rowmax = float64(-np.inf)

    Q_offset = BLOCK_SIZE * cuda.blockIdx.x
    sQ[tid] = Q[cuda.blockIdx.y, cuda.blockIdx.z, 0, tid + Q_offset]
    
    cuda.syncthreads()

    for i in range(SEQ_LEN):
        
        sK[tid] = K[cuda.blockIdx.y, cuda.blockIdx.z, i, tid]
        
        cuda.syncthreads()

        tmp = float64(0)

        for k in range(D_HEAD):
            tmp += sQ[k] * sK[k]

        sS[cuda.blockIdx.x * SEQ_LEN + i] = tmp
        tile_rowmax = cuda.libdevice.fmax(tile_rowmax, sS[cuda.blockIdx.x * SEQ_LEN + i])

        cuda.syncthreads()

    prev_denominator = float64(0)
    tile_denominator = float64(0)
    tile_numerator = float64(0)

    for i in range(SEQ_LEN):
        #FIXME: do we really need to split ?
        tile_numerator = cuda.libdevice.exp(sS[cuda.blockIdx.x * SEQ_LEN + i] - tile_rowmax)
        cuda.syncthreads()
        sS[cuda.blockIdx.x * SEQ_LEN + i] = tile_numerator
        tile_denominator += sS[cuda.blockIdx.x * SEQ_LEN + i]

    cuda.syncthreads()

    new_rowmax = cuda.libdevice.fmax(prev_rowmax, tile_rowmax)
    update_prev_exponent = cuda.libdevice.exp(prev_rowmax - new_rowmax)
    new_denominator = prev_denominator * update_prev_exponent + cuda.libdevice.exp(tile_rowmax - new_rowmax) * tile_denominator

    sO[tid] = O[cuda.blockIdx.y, cuda.blockIdx.z, 0, tid + Q_offset]

    cuda.syncthreads()

    left = sO[tid] * (prev_denominator * update_prev_exponent) / new_denominator
    
    cuda.syncthreads()

    tmp = float64(0)

    for i in range(SEQ_LEN):
        #FIXME: Can we load sK once outside of for loop?
        sK[tid] = V[cuda.blockIdx.y, cuda.blockIdx.z, i, tid]
        right = (sS[cuda.blockIdx.x * SEQ_LEN + i] * cuda.libdevice.exp(tile_rowmax - new_rowmax)) / new_denominator
        tmp += right * sK[tid]
        
    sO[tid] = left + tmp

    cuda.syncthreads()
    
    O[cuda.blockIdx.y, cuda.blockIdx.z, 0, tid + Q_offset] = sO[tid]

    cuda.syncthreads()

    tile_rowmax = new_rowmax
    tile_denominator = new_denominator
     
def flash_attention_forward_gpu(d_Q, d_K, d_V):
    B, C, SEQ_LEN, D_HEAD = d_Q.shape
    dtype = get_torch_dtype(d_Q.dtype.name)

    # Convert torch to numpy dtype

    sizeof = lambda x: torch.tensor(0, dtype=x).element_size()

    O = torch.zeros((B, C, SEQ_LEN, D_HEAD), dtype=torch.float64)
    d_O = cuda.to_device(O)

    stream = cuda.default_stream()
    # TODO: should support multiple blocks (i.e: D_HEAD // 2) 
    threadsperblock = (D_HEAD, 1, 1) # Should be max 1024
    blockspergrid = (math.ceil(d_Q.shape[2] * d_Q.shape[3] / threadsperblock[0]), B, C)
    shared_memory_size = (D_HEAD * sizeof(dtype)) * 3 + (SEQ_LEN**2 * sizeof(dtype))
    # shared memory per block: 49152 bytes
    print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}, shared_memory_size: {shared_memory_size}")

    # _flash_attention_forward_kernel[blockspergrid, threadsperblock, stream, shared_memory_size](
    #     d_Q,
    #     d_K,
    #     d_V,
    #     d_O,
    #     # https://github.com/numba/numba/issues/8883
    #     torch2numpy_dtype(d_Q.dtype.name),
    # )
    
    _flash_attention_forward_kernel[blockspergrid, threadsperblock, stream, shared_memory_size](
        d_Q,
        d_K,
        d_V,
        d_O,
    )

    cuda.synchronize()

    return d_O.copy_to_host()

if __name__ == "__main__":

    #TODO: https://stackoverflow.com/questions/30209088/how-many-blocks-can-be-allocated-if-i-use-shared-memory
    # B, C, SEQ_LEN, D_HEAD = (32, 32, 128, 128) #128
    B, C, SEQ_LEN, D_HEAD = (32, 32, 50, 1024) #128

    Q = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    K = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    V = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10

    d_Q = cuda.to_device(Q)
    d_K = cuda.to_device(K)
    d_V = cuda.to_device(V)

    start_cpu = time.time()
    ref_cpu = ref_attention(Q.clone(), K.clone(), V.clone())
    end_cpu = time.time()

    start_gpu = time.time()
    O = flash_attention_forward_gpu(d_Q, d_K, d_V)
    end_gpu = time.time()

    assert torch.allclose(ref_cpu, torch.from_numpy(O))
    
    print(f"CPU time: {end_cpu - start_cpu} seconds")
    print(f"GPU time: {end_gpu - start_gpu} seconds")