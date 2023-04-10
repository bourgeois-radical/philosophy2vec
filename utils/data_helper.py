from collections import Counter
from typing import Tuple, Dict, List, Union

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def build_sorted_vocabulary(text_as_preprocessed_tokens: List[str], sort_freq: int = 10) -> \
                                                                            Tuple[Dict[str, int], pd.DataFrame, int]:
    """Builds vocabulary filtered by the most frequent words (in descending order).

    Parameters
    ----------
    text_as_preprocessed_tokens : List[str]
    sort_freq : int
        Tokens whose frequency of occurrence in the text is more than sort_freq will be added to the vocabulary.

    Returns
    -------
        vocab_freq : Dict[str, int]
            Vocabulary sorted by frequency in descending order.
        vocab_df : pd.DataFrame
            Vocabulary filtered by sort_freq.
        vocab_len : int
            The length of the filtered vocabulary.
    """
    # create vocab and sort it by frequency
    vocab = dict(Counter(text_as_preprocessed_tokens))
    vocab_freq = {k: v for k, v in sorted(vocab.items(), key=lambda x: x[1], reverse=True)}

    # filter words by frequency > sort_freq
    vocab_df = pd.DataFrame(vocab_freq.items(), columns=['token', 'freq'])
    vocab_df = vocab_df[vocab_df.freq > sort_freq]
    vocab_len = vocab_df.shape[0]

    return vocab_freq, vocab_df, vocab_len


def build_skip_gram_dataset(text_as_preprocessed_tokens: List[str], vocab_df: pd.DataFrame, vocab_len: int,
                            window_size: int = 3) -> Tuple[List[int], List[int], List[str]]:
    """
    Parameters
    ----------
    text_as_preprocessed_tokens : List[str]
        The whole text
    vocab_df : pd.DataFrame
        Vocabulary filtered by sort_freq
    vocab_len : int
        The length of the filtered vocabulary
    window_size : int
        The radius to be used for dataset building

    Returns
    -------
    X : List[int]
    y : List[int]
    vocab_list : List[int]
        The final vocabulary which includes <'UNK'> token

    Notes
    _____
    Building a dataset for Skip-Gram:
        in general:
            let window_size=k, center_word_idx=i then
            X = [i, i, i, i, i, i]; y = [(i-k):i, i:(i+k)]

        example:
            let window_size=3 and center_word_idx=67 then
            X = [67, 67, 67, 67, 67, 67]
            y = [64, 65, 66,    68, 69, 70],
            where each index corresponds to the certain embedding from the embedding layer.

    Building a dataset for CBOW:
        in general:
            let window_size=k, center_word=i then:
            X = arithmetic mean of embeddings with indices: ([(i-k):i, i:(i+k)])
            y = [i]

        example:
            let window_size=3 and center_word_idx=67 then
            X = 1/6 * [64 + 65 + 66 + 68 + 69 + 70]
            y = [67],
            where each index corresponds to the certain embedding from the embedding layer.
    """
    X_as_words = []
    y_as_words = []

    for center_word_idx in range(window_size, len(text_as_preprocessed_tokens) - window_size):

        X_extended = [text_as_preprocessed_tokens[center_word_idx]] * (2 * window_size)
        X_as_words.extend(X_extended)

        for context_index_left in range(center_word_idx - window_size, center_word_idx):
            y_as_words.append(text_as_preprocessed_tokens[context_index_left])

        for context_index_right in range(center_word_idx + 1, window_size + center_word_idx + 1):
            y_as_words.append(text_as_preprocessed_tokens[context_index_right])

    # Before words are passed to the model, they must be encoded as IDs.
    # An ID corresponds to a word index in the vocabulary.
    # Out-of-vocabulary words (= <UNK>) are encoded with the ID=vocab_length.

    # vocab_df already has only the most frequent words
    vocab_list = vocab_df.token.values.tolist()
    X = []
    y = []

    for x_as_word in X_as_words:
        if x_as_word in vocab_list:
            X.append(vocab_list.index(x_as_word))
        else:
            X.append(vocab_len)

    for y_as_word in y_as_words:
        if y_as_word in vocab_list:
            y.append(vocab_list.index(y_as_word))
        else:
            y.append(vocab_len)

    # <UNK>'s ID equals to vocab_length in training data
    vocab_list.append('<UNK>')

    return X, y, vocab_list


class Word2VecDataset(Dataset):
    """DataLoader for PyTorch model"""

    def __init__(self, X: torch.LongTensor, y: Union[torch.LongTensor, np.ndarray]) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.X[idx], self.y[idx]
