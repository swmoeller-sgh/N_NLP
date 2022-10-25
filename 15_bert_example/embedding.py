import torch
from torch import nn, Tensor


class Embedding(nn.Module):
    """
    The embedding is the first layer in BERT that takes the input and creates a lookup table. The parameters of the
    embedding layers are learnable, which means when the learning process is over the embeddings will cluster similar
    inputs (2 concatenated sentences) together.

    The embedding layer also preserves different relationships between words such as: semantic, syntactic, linear,
    and since BERT is bidirectional it will also preserve contextual relationships as well.

    In the case of BERT, it creates three embeddings for
    - Token,
    - Segments and
    - Position.
    add them all up and normalize them via LayerNorm.
    """

    def __init__(self, vocab_size: int, d_model: int, maxlen: int, n_segments: int = 2):
        super(Embedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # token embedding
        self.position_embedding = nn.Embedding(maxlen, d_model)  # position embedding
        self.segment_embedding = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: Tensor, segment_ids: Tensor) -> Tensor:
        """
        
        Args:
            input_ids: a tensor of shape (batchsize, maxlen) containing the ids of the words in each Batch entry (2 concated sentences)
            segment_ids: a tensor of shape (batchsize, maxlen) containing the ids of the sentence the words belong to.

        Returns:
            an embedding tensor of shape (batchsize, maxlen, d_model)
        """
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(input_ids)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.token_embedding(input_ids) + self.position_embedding(pos) + self.segment_embedding(segment_ids)
        return self.norm(embedding)
