from typing import Tuple

from torch import Tensor
from torch.nn import Module

from basic_mlp import MLP
from multihead_attention import MultiHeadAttention


class EncoderLayer(Module):
    """
    The encoder has two main components:

    - Multi-head Attention
    - Position-wise feed-forward network.
    The work of the encoder is to find representations and patterns from the input and attention mask.
    """
    def __init__(self, d_model: int):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model=d_model)
        # self.pos_ffn = PoswiseFeedForwardNet()
        self.pos_ffn = MLP(input_size=d_model, output_size=d_model, num_layers=2, hidd_size=8)

    def forward(self, enc_inputs: Tensor, enc_self_attn_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        
        Args:
            enc_inputs: a tensor of shape (batchsize, maxlen, d_model) containing an encoding of each word of a sentence
                this encoding probably already went through some encoder layers. 
            enc_self_attn_mask: a tensor of shape (batchsize, maxlen, maxlen) containing the encoded self attention mask

        Returns:
            again an encoding of shape (batchsize, maxlen, d_model) and the computed attention.
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
