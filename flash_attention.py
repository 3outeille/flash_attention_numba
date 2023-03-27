import random
import os
import numpy as np
import torch
from numba import cuda, float64
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

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of BLOCKSIZExBLOCKSIZE elements.
# BLOCKSIZE should not be larger than 32 in this example (because max threads per block is 1024, and 32x32=1024)
BLOCKSIZE = 4 # (16 * 16) * 4 = 1024 (because we have QKVO in shared memory)


@cuda.jit
def flash_attention_forward_gpu(Q, K, V, O, tile_rowmax, tile_numerator, tile_denominator, new_rowmax, state_rowmax, state_denominator, right, right_V, left):
    
    # TODO: (seq_len, d_head) = (2,2)
    # 1. Test on 1 tile (BLOCKSIZE = 4)
    # 2. Test on 2 tiles (BLOCKSIZE = 2)

    sQ = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float64)
    sK = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float64)
    sV = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float64)
    sO = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=float64)

    col = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    row = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Matmul sQ @ sK
    #FIXME: Are we filling tiles left to right or top to bottom ?
    for blockId in range(cuda.gridDim.x):

        # Load a tile of A and B into shared memory
        if row < Q.shape[0] and tx + blockId * BLOCKSIZE < Q.shape[1]:
            sQ[ty, tx] = Q[row, tx + blockId * BLOCKSIZE]
        if col < K.shape[1] and ty + blockId * BLOCKSIZE < K.shape[0]:
            #FIXME: K should be transposed here (S). Can we just do .transpose() ?
            sK[ty, tx] = K[ty + blockId * BLOCKSIZE, col]
        if col < V.shape[1] and ty + blockId * BLOCKSIZE < V.shape[0]:
            sV[ty, tx] = V[ty + blockId * BLOCKSIZE, col]

        cuda.syncthreads()

        # Matmul on the current tile
        for k in range(BLOCKSIZE):
            #FIXME: K should be transposed here (S). Can we just do .transpose() ?
            sO[ty, tx] += sQ[ty, k] * sK[k, tx]

        cuda.syncthreads()
        
        # find max of each row of the current loaded block
        cuda.atomic.max(tile_rowmax, row, sO[ty, tx])

    # Online softmax
    cuda.syncthreads() # TODO: is this necessary?

    # compute the softmax numerator (float64)
    tile_numerator[ty, tx] = cuda.libdevice.exp(sO[ty, tx] - tile_rowmax[row])

    cuda.syncthreads()
    
    # Compute the softmax denominator
    cuda.atomic.add(tile_denominator, row, tile_numerator[ty, tx])

    # Compute new rowmax
    cuda.atomic.max(new_rowmax, row, tile_rowmax[row])

    #TODO: Use variable instead of array ?
    new_denominator = state_denominator[row] * (cuda.libdevice.exp(state_rowmax[row] - new_rowmax[row])) + cuda.libdevice.exp(tile_rowmax[row] - new_rowmax[row]) * tile_denominator[row]
    cuda.syncthreads()

    if row < Q.shape[0] and col < K.shape[1]:
        left[ty, tx] = O[row, col] * (state_denominator[row] * cuda.libdevice.exp(state_rowmax[row] - new_rowmax[row])) / new_denominator 
        right[ty, tx] = (tile_numerator[ty, tx] * cuda.libdevice.exp(tile_rowmax[row] - new_rowmax[row])) / new_denominator
        cuda.syncthreads()

        # right @ sV
        for k in range(BLOCKSIZE):
            right_V[ty, tx] += right[ty, k] * sV[k, tx]

        cuda.syncthreads()

        O[row, col] = left[row, col] + right_V[row, col]

    cuda.syncthreads()

    state_rowmax[row] = new_rowmax[row]
    state_denominator[row] = new_denominator

if __name__ == "__main__":
    seed_everything()

    seq_len = 4 # 32
    d_head = 4 # 32

    # h_Q = (torch.arange(seq_len * d_head, dtype=torch.float64).reshape(seq_len, d_head) + 1.)
    # h_K = (torch.arange(seq_len * d_head, dtype=torch.float64).reshape(seq_len, d_head).T + 1.)
    # h_V = (torch.arange(seq_len * d_head, dtype=torch.float64).reshape(seq_len, d_head) + 1.)
    # h_state_rowmax = torch.full((seq_len, 1), -torch.inf) # Row max "m"
    # h_state_denominator = torch.zeros((seq_len, 1)) # Softmax denominator "l"

    h_Q = (torch.arange(seq_len * d_head, dtype=torch.float64).reshape(seq_len, d_head) + 1.) / 10
    h_K = (torch.arange(seq_len * d_head, dtype=torch.float64).reshape(seq_len, d_head) + 1.) / 10
    h_V = (torch.arange(seq_len * d_head, dtype=torch.float64).reshape(seq_len, d_head) + 1.) / 10
    h_actual = torch.zeros((seq_len, d_head), dtype=torch.float64)

    expected = flash_attention_forward_cpu(h_Q.clone(), h_K.clone(), h_V.clone())
    expected = expected.to(torch.float64)

    d_Q = cuda.to_device(h_Q)
    d_K = cuda.to_device(h_K.T) #FIXME: For now, we transpose K outside the kernel
    d_V = cuda.to_device(h_V)
    d_actual = cuda.to_device(h_actual)

    threadsperblock = (BLOCKSIZE, BLOCKSIZE)
    blockspergrid_x = math.ceil(h_actual.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(h_actual.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(f"blockspergrid: {blockspergrid}, threadsperblock: {threadsperblock}")

    # Allocation
    h_tile_rowmax = torch.zeros(seq_len, dtype=torch.float64)
    d_tile_rowmax = cuda.to_device(h_tile_rowmax)

    h_tile_numerator = torch.zeros(seq_len * d_head, dtype=torch.float64).reshape(seq_len, d_head)
    d_tile_numerator = cuda.to_device(h_tile_numerator)
    h_tile_denominator = torch.empty(seq_len, dtype=torch.float64).fill_(0.)
    d_tile_denominator = cuda.to_device(h_tile_denominator)

    h_state_rowmax = torch.empty(seq_len).fill_(-float('inf'))
    d_state_rowmax = cuda.to_device(h_state_rowmax)

    h_new_rowmax = torch.empty(seq_len).fill_(-float('inf'))
    d_new_rowmax = cuda.to_device(h_new_rowmax)

    h_state_denominator = torch.zeros(seq_len, dtype=torch.float64)
    d_state_denominator = cuda.to_device(h_state_denominator)

    h_right = torch.zeros(BLOCKSIZE * BLOCKSIZE, dtype=torch.float64).reshape(BLOCKSIZE, BLOCKSIZE)
    d_right = cuda.to_device(h_right)

    h_right_V = torch.zeros(BLOCKSIZE * d_head, dtype=torch.float64).reshape(BLOCKSIZE, d_head)
    d_right_V = cuda.to_device(h_right_V)

    h_left = torch.zeros(BLOCKSIZE * d_head, dtype=torch.float64).reshape(BLOCKSIZE, d_head)
    d_left = cuda.to_device(h_left)

    # TODO: K must be transposed before passing it to the kernel (for Attention)
    flash_attention_forward_gpu[blockspergrid, threadsperblock](
        d_Q,
        d_K,
        d_V,
        d_actual,
        d_tile_rowmax,
        d_tile_numerator,
        d_tile_denominator,
        d_new_rowmax,
        d_state_rowmax,
        d_state_denominator,
        d_right,
        d_right_V,
        d_left
    )

    h_actual = torch.from_numpy(d_actual.copy_to_host())

    h_tile_rowmax = torch.from_numpy(d_tile_rowmax.copy_to_host())
    print(h_tile_rowmax)

    # print("--- tile_numerator ---")
    # h_tile_numerator = torch.from_numpy(d_tile_numerator.copy_to_host())
    # print(h_tile_numerator)
    
    # print("--- tile_denominator ---")
    # h_tile_denominator = torch.from_numpy(d_tile_denominator.copy_to_host())
    # print(h_tile_denominator)

    # print("--- state denominator---")
    # h_state_denominator = torch.from_numpy(d_state_denominator.copy_to_host())
    # print(h_state_denominator)
    print("--- left ---")
    h_left = torch.from_numpy(d_left.copy_to_host())
    print(h_left)

    print("--- right ---")
    h_right = torch.from_numpy(d_right.copy_to_host())
    print(h_right)

    print("--- V ---")
    print(h_V)

    print("--- right_V ---")
    h_right_V = torch.from_numpy(d_right_V.copy_to_host())
    print(h_right_V)


    print("--- actual ---")
    print(h_actual)

    print("--- expected ---")
    print(expected, type(expected))

    assert torch.allclose(expected, h_actual)