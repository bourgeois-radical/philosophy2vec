# built-in modules and packages
from typing import Tuple

# installed modules and packages
import torch.nn as nn


class Word2VecModel(nn.Module):
    """PyTorch Word2Vec model consists of two layers: Embedding and Linear
         and provides two architectures: Skip-Gram and CBOW"""

    def __init__(self, vocab_len: int, model_type: str = 'skipgram', embd_dim: int = 300, embd_max_norm: int = 1) \
            -> None:
        """Hyperparameters of the model

        Parameters
        ----------
        vocab_len : int
            The length of the vocabulary filtered by certain frequency.
        model_type : str
            Word2Vec architecture: 'skipgram' or 'cbow'.
        embd_dim : int
            Dimensionality of an embedding.
        embd_max_norm : int
            Works as a regularization parameter and prevents weights in Embedding Layer grow uncontrollably.
            Equals to '1' by default.
        """
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_len,
            embedding_dim=embd_dim,
            max_norm=embd_max_norm,
        )
        self.linear = nn.Linear(
            in_features=embd_dim,
            out_features=vocab_len,
        )
        self.model_type = model_type

    def forward(self, inputs_: Tuple[int, int]):
        x = self.embeddings(inputs_)
        if self.model_type == 'cbow':
            x = x.mean(axis=1)
        x = self.linear(x)
        return x
