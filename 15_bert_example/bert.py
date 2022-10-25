import math

import torch
from torch import nn, LongTensor

from attention_pad_mask import get_attn_pad_mask
from embedding import Embedding
from encoding_layer import EncoderLayer


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BERT(nn.Module):
    def __init__(self, maxlen: int, n_layers: int = 6, d_model: int = 512, vocab_size: int = 30):
        """
        
        Args:
            maxlen: the maximal length of the time series (i.e. the longest sentence in the data)
            n_layers: the number of encoder layers
            d_model: the size of the embedding vectors used by the model.  
            vocab_size: the number of different words 
        """
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model, maxlen=maxlen)
        self.layers = nn.ModuleList([EncoderLayer(d_model) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.token_embedding.weight
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids: LongTensor, segment_ids: LongTensor, masked_pos: LongTensor):
        """

        Args:
            input_ids: a tensor of shape (batchsize, maxlen) containing the ids of the words in each Batch entry (2 concated sentences)
            segment_ids: a tensor of shape (batchsize, maxlen) containing the ids of the sentence the words belong to.
            masked_pos:

        Returns:

        """
        output = self.embedding(input_ids, segment_ids)  # (batchsize, maxlen, d_model)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # (batchsize, maxlen, maxlen)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_model, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))  # [batch_size, max_pred, d_model]

        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias  # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf
