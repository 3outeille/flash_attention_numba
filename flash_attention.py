import random
import os
import numpy as np
import torch
from numba import cuda, float32
import math


def seed_everything():
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def ref_attention(Q, K, V):
    #TODO: divide by sqrt(d_k)
    return torch.softmax(Q @ K.T, dim=1) @ V

def flash_attention_forward_cpu(Q, K, V):
    """
        No mask + No dropout version
    """
    #TODO: divide by sqrt(d_k)

    seq_len, d_head = Q.shape
    # TODO: Should be size of tile (Shared memory size / 2)
    # TODO: Becareful, in GPU, we use the same shared memory for QKV and O (output) so we should divide Shared memory by 4
    Br = 4 # 4 # Control number of row loaded
    Bc = 2 #d_head # Control number of column loaded

    # line 3
    O = torch.zeros((seq_len, d_head))
    state_rowmax = torch.full((seq_len, 1), -torch.inf) # Row max "m"
    state_denominator = torch.zeros((seq_len, 1)) # Softmax denominator "l"

    # This is range(seq_len) because we load part of rows with ALL its columns
    for block_start_Bc in range(0, seq_len, Bc):
        
        block_end_Bc =block_start_Bc + Bc
        
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
            Oi = (Oi * (prev_denominator * update_prev_exponent) / new_denominator) + ((tile_numerator * torch.exp(tile_rowmax - new_rowmax)) / new_denominator) @ Vj

            # line 16: save statistics
            state_rowmax[block_start_Br:block_end_Br, :] = new_rowmax
            state_denominator[block_start_Br:block_end_Br, :] = new_denominator 

            O[block_start_Br:block_end_Br, :] = Oi
    return O

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of BLOCKSIZExBLOCKSIZE elements.
# BLOCKSIZE should not be larger than 32 in this example (because max threads per block is 1024, and 32x32=1024)
BLOCKSIZE = 16 # (16 * 16) * 4 = 1024 (because we have QKVO in shared memory)

@cuda.jit
def flash_attention_forward_gpu(Q, K, V, O):
    
    # TODO:
    # 1. Load equally in shared memory for Q, K, V, O [DONE]
    # 2. Just perform matmul between Q,K to check if everything still work [DONE]
    # 3. Adapt with online softmax 


    sQ = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float32)
    sK = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float32)
    # sV = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float32)
    sO = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float32)

    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    tmp = float32(0.)

    # Loop until all tiles of a given direction are processed 
    # (in order to have a complete matrix multiplication along this direction)
    for blockId in range(cuda.gridDim.x):

        # Load a tile of A and B into shared memory
        if row < Q.shape[0] and tx + blockId * BLOCKSIZE < Q.shape[1]:
            sQ[ty, tx] = Q[row, tx + blockId * BLOCKSIZE]
        if col < K.shape[1] and ty + blockId * BLOCKSIZE < K.shape[0]:
            sK[ty, tx] = K[ty + blockId * BLOCKSIZE, col]
                    
        cuda.syncthreads()

        for k in range(BLOCKSIZE):
            tmp += sQ[ty, k] * sK[k, tx]

        cuda.syncthreads()

    sO[ty, tx] = tmp

    cuda.syncthreads()

    if row < Q.shape[0] and col < K.shape[1]:
        O[row, col] = sO[ty, tx]


if __name__ == "__main__":
    seed_everything()

    seq_len = 6 # 32
    d_head = 6 # 32

    h_Q = (torch.arange(seq_len * d_head, dtype=torch.float32).reshape(seq_len, d_head) + 1.)
    h_K = (torch.arange(seq_len * d_head, dtype=torch.float32).reshape(seq_len, d_head) + 1.)
    h_V = (torch.arange(seq_len * d_head, dtype=torch.float32).reshape(seq_len, d_head) + 1.)

    # h_Q = (torch.arange(seq_len * d_head, dtype=torch.float32).reshape(seq_len, d_head) + 1.) / 10
    # h_K = (torch.arange(seq_len * d_head, dtype=torch.float32).reshape(seq_len, d_head) + 1.) / 10
    # h_V = (torch.arange(seq_len * d_head, dtype=torch.float32).reshape(seq_len, d_head) + 1.) / 10
    h_actual = torch.zeros((seq_len, d_head), dtype=torch.float32)

    # expected = ref_attention(h_Q.clone(), h_K.clone(), h_V.clone())

    d_Q = cuda.to_device(h_Q)
    d_K = cuda.to_device(h_K)
    d_V = cuda.to_device(h_V)
    d_actual = cuda.to_device(h_actual)

    threadsperblock = (BLOCKSIZE, BLOCKSIZE)
    blockspergrid_x = math.ceil(h_actual.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(h_actual.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}")

    flash_attention_forward_gpu[blockspergrid, threadsperblock](d_Q, d_K, d_V, d_actual)

    h_actual = torch.from_numpy(d_actual.copy_to_host())
    print(h_Q @ h_K)
    print(h_actual)
    print((h_Q @ h_K) - h_actual)

    # assert torch.allclose(expected, h_actual)