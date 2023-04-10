import torch.nn as nn
from typing import Tuple


class Word2VecModel(nn.Module):
    """PyTorch Word2Vec model consisting of two layers: embedding and linear"""

    def __init__(self, vocab_size: int, model_type: str = 'skipgram', embd_dim: int = 300, embd_max_norm: int = 1) \
                                                                                                                -> None:
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embd_dim,
            max_norm=embd_max_norm,
        )
        self.linear = nn.Linear(
            in_features=embd_dim,
            out_features=vocab_size,
        )
        self.model_type = model_type

    def forward(self, inputs_: Tuple[int, int]):
        x = self.embeddings(inputs_)
        if self.model_type == 'cbow':
            x = x.mean(axis=1)
        x = self.linear(x)
        return x
