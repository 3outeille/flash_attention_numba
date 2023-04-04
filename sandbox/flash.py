import numpy as np
from numba import cuda, float64, int64
import math
import torch

def ref_attention(Q, K, V):
    S = torch.matmul(Q, K.transpose(-2, -1))
    P = torch.softmax(S, dim=-1)
    O =  torch.matmul(P, V)
    return O

def flash_attention_forward_cpu(Q, K, V):
    """
        No mask + No dropout version
    """
    #TODO: divide by sqrt(d_k)

    seq_len, d_head = Q.shape
    # TODO: Should be size of tile (Shared memory size / 2)
    # TODO: Becareful, in GPU, we use the same shared memory for QKV and O (output) so we should divide Shared memory by 4
    Br = Q.shape[-2]  # 4 # Control number of row loaded
    Bc = Q.shape[-1] #d_head # Control number of column loaded

    # line 3
    O = torch.zeros((seq_len, d_head), dtype=Q.dtype)
    state_rowmax = torch.full((seq_len, 1), -torch.inf) # Row max "m"
    state_denominator = torch.zeros((seq_len, 1), dtype=Q.dtype) # Softmax denominator "l"

    # This is range(seq_len) because we load part of rows with ALL its columns
    for block_start_Bc in range(0, seq_len, Bc):
        
        block_end_Bc = block_start_Bc + Bc
        
        # line 4
        Kj = K[block_start_Bc:block_end_Bc, :]  # shape Bc x d_head
        Vj = V[block_start_Bc:block_end_Bc, :]  # shape Bc x d_head
        
        for block_start_Br in range(0, seq_len, Br):
            
            block_end_Br = block_start_Br + Br

            # line 4,5,9
            prev_rowmax = state_rowmax[block_start_Br:block_end_Br, :]  # shape Br x 1
            prev_denominator = state_denominator[block_start_Br:block_end_Br, :]  # shape Br x 1
            Oi = O[block_start_Br:block_end_Br, :]  # shape Br x d_head
            Qi = Q[block_start_Br:block_end_Br, :]  # shape Br x d_head

            # line 10
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1))  # shape Br x Bc
            
            print("=================================================")
            # line 12: find max of each row of the current loaded block
            tile_rowmax = torch.max(Sij, dim=-1).values[..., None]
            tile_numerator = torch.exp(Sij - tile_rowmax)
            print("--- tile_rowmax ---")
            print(tile_rowmax)
            # line 12: compute the softmax numerator
            print("--- tile_numerator ---")
            print(tile_numerator)
            # line 12: compute the softmax denominator
            tile_denominator = torch.sum(tile_numerator, dim=-1)[:, None]
            print("--- tile_denominator ---")
            print(tile_denominator)

            # line 13: Online softmax
            new_rowmax = torch.max(torch.column_stack([prev_rowmax, tile_rowmax]), dim=-1).values[..., None]
            print("--- new_rowmax ---")
            print(new_rowmax)

            update_prev_exponent = torch.exp(prev_rowmax - new_rowmax)
            new_denominator = prev_denominator * update_prev_exponent + torch.exp(tile_rowmax - new_rowmax) * tile_denominator

            # line 15: Attention computation on tile
            # Oi = [exp(Q_i @ Ki.T - curr_rowmax) / sum(exp(Q_i @ Ki.T - curr_rowmax))] @ Vi
            left = (Oi * (prev_denominator * update_prev_exponent) / new_denominator)
            right = ((tile_numerator * torch.exp(tile_rowmax - new_rowmax)) / new_denominator)
            print("---left---")
            print(left)
            print("---right---")
            print(right, right.shape)
            print("---V---")
            print(Vj, Vj.shape)
            Oi = left + torch.matmul(right, Vj)
            print("----right V----")
            print(right @ Vj)
            # line 16: save statistics
            state_rowmax[block_start_Br:block_end_Br, :] = new_rowmax
            state_denominator[block_start_Br:block_end_Br, :] = new_denominator 

            O[block_start_Br:block_end_Br, :] = Oi
            # print("---O---")
            # print(O[block_start_Br:block_end_Br, :])
            
    return O


@cuda.jit
def flash_attention(Q, K, V, O, S_debug):

    # TODO: Handle multiple blocks
    # TODO: Handle K transpose loading
    # TODO: divide by sqrt(dk)

    SEQ_LEN = Q.shape[2]
    D_HEAD = Q.shape[3]
    BLOCK_SIZE = cuda.blockDim.x
    tid = cuda.threadIdx.x

    sQ = cuda.shared.array(0, dtype=np.float64)[:D_HEAD]
    sK = cuda.shared.array(0, dtype=np.float64)[D_HEAD:2*D_HEAD]
    sO = cuda.shared.array(0, dtype=np.float64)[2*D_HEAD:3*D_HEAD]
    sS = cuda.shared.array(0, dtype=np.float64)[3*D_HEAD:3*D_HEAD + SEQ_LEN**2]

    if tid >= D_HEAD:
        return

    prev_rowmax = float64(-np.inf)
    tile_rowmax = float64(-np.inf)

    Q_offset = BLOCK_SIZE * cuda.blockIdx.x
    sQ[tid] = Q[cuda.blockIdx.y, cuda.blockIdx.z, 0, tid + Q_offset]

    for i in range(SEQ_LEN):
        sK[tid] = K[cuda.blockIdx.y, cuda.blockIdx.z, i, tid]
        
        cuda.syncthreads()

        tmp = float64(0)

        for k in range(D_HEAD):
            tmp += sQ[k] * sK[k]

        sS[cuda.blockIdx.x * SEQ_LEN + i] = tmp
        S_debug[cuda.blockIdx.y, cuda.blockIdx.z, cuda.blockIdx.x, i] = sS[cuda.blockIdx.x * SEQ_LEN + i]
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

    cuda.syncthreads()

    update_prev_exponent = cuda.libdevice.exp(prev_rowmax - new_rowmax)
    
    cuda.syncthreads()

    new_denominator = prev_denominator * update_prev_exponent + cuda.libdevice.exp(tile_rowmax - new_rowmax) * tile_denominator

    cuda.syncthreads()

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
     

if __name__ == "__main__":
    B, C, SEQ_LEN, D_HEAD = (3, 4, 2, 3) #128
    Q = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    K = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    V = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=torch.float64).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    # TODO: init O with zeros inside the kernel
    O = torch.zeros((B, C, SEQ_LEN, D_HEAD), dtype=torch.float64)

    # ref = ref_attention(Q.clone()[0, 0, ...], K.clone()[0, 0, ...], V.clone()[0, 0, ...])
    # pred = flash_attention_forward_cpu(Q.clone()[0, 0, ...], K.clone()[0, 0, ...], V.clone()[0, 0, ...])

    # assert torch.allclose(ref, pred)

    d_Q = cuda.to_device(Q)
    d_K = cuda.to_device(K)
    d_V = cuda.to_device(V)
    d_O = cuda.to_device(O)

    # TODO: Find best number of threads and blocks given compute capability given MAX Shareed Memory size for my gpu 
    stream = cuda.default_stream()
    threadsperblock = (D_HEAD, 1, 1)
    blockspergrid = (math.ceil(Q.shape[2] * Q.shape[3] / threadsperblock[0]), B, C)
    shared_memory_size = (D_HEAD * np.dtype(np.float64).itemsize) * 3 + (SEQ_LEN**2 * np.dtype(np.float64).itemsize)
    print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}, shared_memory_size: {shared_memory_size}")

    # ====== DEBUG ======
    d_S_debug = cuda.to_device(torch.zeros((B, C, SEQ_LEN, SEQ_LEN), dtype=torch.float64))

    flash_attention[blockspergrid, threadsperblock, stream, shared_memory_size](
        d_Q,
        d_K,
        d_V,
        d_O,
        # DEBUG
        d_S_debug,
    )
    
    cuda.synchronize()

    # S_debug = torch.from_numpy(d_S_debug.copy_to_host())
    O = torch.from_numpy(d_O.copy_to_host())
    ref_cpu = ref_attention(Q.clone(), K.clone(), V.clone())
    print(O - ref_cpu)

    assert torch.allclose(ref_cpu, O)