import pytest
import random
import os
import math
import numpy as np
import torch
from numba import cuda

from flash_attention import ref_attention, flash_attention_forward_gpu

class TestFlashAttention:
        
    @pytest.fixture(scope="session")
    def seed_everything(self):
        seed = 42
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    

    def initialize_matrices(self, b, c, seq_len, d_head, TORCH_DTYPE):
        Q = torch.randn(b, c, seq_len, d_head, dtype=TORCH_DTYPE)
        K = torch.randn(b, c, seq_len, d_head, dtype=TORCH_DTYPE)
        V = torch.randn(b, c, seq_len, d_head, dtype=TORCH_DTYPE)
        return Q, K, V

    def copy_to_gpu(self, Q, K, V):
        d_Q = cuda.to_device(Q)
        d_K = cuda.to_device(K)
        d_V = cuda.to_device(V)
        return d_Q, d_K, d_V

    @pytest.mark.usefixtures("seed_everything")
    @pytest.mark.parametrize("b", [32])
    @pytest.mark.parametrize("c", [32])
    @pytest.mark.parametrize("seq_len", [32, 64, 128])
    @pytest.mark.parametrize("d_head", [32, 64, 128])
    def test_forward(self, b, c, seq_len, d_head):
        print(f"b: {b}, c: {c}, seq_len: {seq_len}, d_head: {d_head}")
        Q, K, V = self.initialize_matrices(b, c, seq_len, d_head, torch.float64)
        d_Q, d_K, d_V = self.copy_to_gpu(Q, K, V)
        expected = ref_attention(Q, K, V)

        actual = flash_attention_forward_gpu(d_Q, d_K, d_V)

        torch.allclose(expected, torch.from_numpy(actual)) 
        
        del Q, K, V, d_Q, d_K, d_V, expected, actual
        torch.cuda.empty_cache()       

    @pytest.mark.skip(reason="Not implemented yet")
    @pytest.mark.usefixtures("seed_everything")
    def test_backward(self):
        pass