import torch

def ref_attention(Q, K, V):
    return torch.softmax(Q @ K.T, dim=1) @ V

def flash_attention_forward(Q, K, V):
    return torch.softmax(Q @ K.T, dim=1) @ V