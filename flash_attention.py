import random
import os
import numpy as np
import torch

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

if __name__ == "__main__":
    seed_everything()

    seq_len = 4
    d_head = 4

    # Q = torch.randn(seq_len, d_head)
    # K = torch.randn(seq_len, d_head)
    # V = torch.randn(seq_len, d_head)

    Q = (torch.arange(seq_len * d_head).reshape(seq_len, d_head) + 1.) / 10
    K = (torch.arange(seq_len * d_head).reshape(seq_len, d_head) + 1.) / 10
    V = (torch.arange(seq_len * d_head).reshape(seq_len, d_head) + 1.) / 10

    expected = ref_attention(Q.clone(), K.clone(), V.clone())
    actual = flash_attention_forward_cpu(Q.clone(), K.clone(), V.clone())
    
    assert torch.allclose(expected, actual)