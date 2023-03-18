import pytest
import random
import os
import numpy as np
import torch

from flash_attention import ref_attention, flash_attention_forward

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

    def initialize_matrices(self, seq_len, d_head):
        return torch.randn(seq_len, d_head), torch.randn(seq_len, d_head), torch.randn(seq_len, d_head)

    @pytest.mark.usefixtures("seed_everything")
    @pytest.mark.parametrize("seq_len", [2, 4, 8, 16, 32, 64])
    @pytest.mark.parametrize("d_head", [2, 4, 8, 16, 32, 64])
    def test_forward(self, seq_len, d_head):
        Q, K, V = self.initialize_matrices(seq_len, d_head)
        expected = ref_attention(Q, K, V)
        actual = flash_attention_forward(Q, K, V)
        torch.allclose(expected, actual)        

    @pytest.mark.skip(reason="Not implemented yet")
    @pytest.mark.usefixtures("seed_everything")
    def test_backward(self):
        pass