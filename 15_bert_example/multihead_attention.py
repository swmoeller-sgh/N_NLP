from typing import Tuple

from scaled_dot_product_attention import ScaledDotProductAttention
from torch import nn, Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int = 64, d_v: int = 64, n_heads: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.W_0 = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.scaled_dotproduct_attention = ScaledDotProductAttention(self.d_k)

    def forward(self, q_input, k_input, v_input, attn_mask) -> Tuple[Tensor, Tensor]:
        """
        the multi-head attentions receives some inputs to generate q, k and v from. These are passed through simple
        linear layers to obtain q, k and v. Then the attention score of the three is calculated.
        This is linearly transformed too to obtain the output of the layer.

        Notes:
            there is a skip connection for the q_input.

        Args:
            q_input: a tensor of shape (batchsize, maxlen, d_model) containing the input for the query
            k_input: a tensor of shape (batchsize, maxlen, d_model) containing the input for the key
            v_input: a tensor of shape (batchsize, maxlen, d_model) containing the input for the value
            attn_mask: 

        Returns:
            - a linear transform of the attention values of shape (batchsize, maxlen, d_model)
            - the computed attention (for debugging/visualization only)
        """
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = q_input, q_input.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q = self.W_Q(q_input).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k = self.W_K(k_input).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v = self.W_V(v_input).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        _, context, attn = self.scaled_dotproduct_attention(q, k, v, attn_mask)
        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.W_0(context)
        return self.norm(output + residual), attn  # output: (batch_size, len_q, d_model)
