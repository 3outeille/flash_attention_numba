import numpy as np
from numba import cuda, float64, int64
import math
import torch

def ref_attention(Q, K, V):
    #TODO: divide by sqrt(d_k)
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
    Br = Q.shape[0] // 2 # 4 # Control number of row loaded
    Bc = Q.shape[1] // 2 #d_head # Control number of column loaded

    # line 3
    O = torch.zeros((seq_len, d_head))
    state_rowmax = torch.full((seq_len, 1), -torch.inf) # Row max "m"
    state_denominator = torch.zeros((seq_len, 1)) # Softmax denominator "l"

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
            Sij = Qi @ Kj.T  # shape Br x Bc
            
            # line 12: find max of each row of the current loaded block
            tile_rowmax = torch.max(Sij, dim=1).values[:, None]
            # line 12: compute the softmax numerator
            tile_numerator = torch.exp(Sij - tile_rowmax)
            # line 12: compute the softmax denominator
            tile_denominator = torch.sum(tile_numerator, dim=1)[:, None]

            # line 13: Online softmax
            new_rowmax = torch.max(torch.column_stack([prev_rowmax, tile_rowmax]), dim=1).values[:, None]
            update_prev_exponent = torch.exp(prev_rowmax - new_rowmax)
            new_denominator = prev_denominator * update_prev_exponent + torch.exp(tile_rowmax - new_rowmax) * tile_denominator

            # line 15: Attention computation on tile
            # Oi = [exp(Q_i @ Ki.T - curr_rowmax) / sum(exp(Q_i @ Ki.T - curr_rowmax))] @ Vi
            left = (Oi * (prev_denominator * update_prev_exponent) / new_denominator)
            right = ((tile_numerator * torch.exp(tile_rowmax - new_rowmax)) / new_denominator)
            print("=================================================")
            print("---left---")
            print(left)
            print("---right---")
            print(right)
            print("---V---")
            print(Vj)
            Oi = left + right @ Vj
            print("----right V----")
            print(right @ Vj)
            # line 16: save statistics
            state_rowmax[block_start_Br:block_end_Br, :] = new_rowmax
            state_denominator[block_start_Br:block_end_Br, :] = new_denominator 

            O[block_start_Br:block_end_Br, :] = Oi
            print("---O---")
            print(O[block_start_Br:block_end_Br, :])
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
    
    sQ = cuda.shared.array(0, dtype=NP_DTYPE)[:D_HEAD]
    sK = cuda.shared.array(0, dtype=NP_DTYPE)[D_HEAD:2*D_HEAD]
    sO = cuda.shared.array(0, dtype=NP_DTYPE)[2*D_HEAD:3*D_HEAD]
    sS = cuda.shared.array(0, dtype=NP_DTYPE)[3*D_HEAD:3*D_HEAD + SEQ_LEN**2]

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

    # TODO: can we remove some syncthreads ?
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
     
    
def get_dtype(type):
    if type == "float32":
        torch_dtype = torch.float32
        np_dtype = np.float32
    elif type == "float64":
        torch_dtype = torch.float64
        np_dtype = np.float64
    elif type == "float16":
        torch_dtype = torch.float16
        np_dtype = np.float16
    else:
        raise NotImplementedError("dtype not supported")
    return torch_dtype, np_dtype

if __name__ == "__main__":

    TORCH_DTYPE, NP_DTYPE = get_dtype("float32")

    #TODO: https://stackoverflow.com/questions/30209088/how-many-blocks-can-be-allocated-if-i-use-shared-memory
    B, C, SEQ_LEN, D_HEAD = (1, 1, 50, 1024) #128
    Q = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=TORCH_DTYPE).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    K = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=TORCH_DTYPE).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    V = (torch.arange(B * C * SEQ_LEN * D_HEAD, dtype=TORCH_DTYPE).reshape(B, C, SEQ_LEN, D_HEAD) + 1.) / 10
    # TODO: init O with zeros inside the kernel
    O = torch.zeros((B, C, SEQ_LEN, D_HEAD), dtype=TORCH_DTYPE)

    # ref = ref_attention(Q.clone()[0, 0, ...], K.clone()[0, 0, ...], V.clone()[0, 0, ...])
    # pred = flash_attention_forward_cpu(Q.clone()[0, 0, ...], K.clone()[0, 0, ...], V.clone()[0, 0, ...])

    # assert torch.allclose(ref, pred)

    d_Q = cuda.to_device(Q)
    d_K = cuda.to_device(K)
    d_V = cuda.to_device(V)
    d_O = cuda.to_device(O)

    stream = cuda.default_stream()
    # TODO: should support multiple blocks (i.e: D_HEAD // 2) 
    threadsperblock = (D_HEAD, 1, 1) # Should be max 1024
    blockspergrid = (math.ceil(Q.shape[2] * Q.shape[3] / threadsperblock[0]), B, C)
    shared_memory_size = (D_HEAD * np.dtype(NP_DTYPE).itemsize) * 3 + (SEQ_LEN**2 * np.dtype(NP_DTYPE).itemsize)
    # shared memory per block: 49152 bytes
    print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}, shared_memory_size: {shared_memory_size}")

    # ====== DEBUG ======
    d_S_debug = cuda.to_device(torch.zeros((B, C, SEQ_LEN, SEQ_LEN), dtype=TORCH_DTYPE))

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