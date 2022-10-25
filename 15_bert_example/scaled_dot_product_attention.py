from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int = 64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attn_mask) -> Tuple[Tensor, Tensor, Tensor]:
        """
        a value is computed for each 
 
        Args:
            q: a tensor of shape (batchsize, n_heads, maxlen, d_k) containing
            k: a tensor of shape (batchsize, n_heads, maxlen, d_k) containing
            v: a tensor of shape (batchsize, n_heads, maxlen, d_v) containing
            attn_mask: a tensor of shape (batchsize, n_heads, maxlen, maxlen) containing
 
        Returns:
            - the computed score (just for debugging/visualization purposes)
            - the context which contains the actual return value
            - the computed attention (just for debugging/visualization purposes)
        """
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)  # (batchsize, n_heads, maxlen, d_v)
        return scores, context, attn
