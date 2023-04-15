# built-in modules and packages
from typing import Tuple, Dict, List

# installed modules and packages
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

# local modules and packages
import utils.model
from utils.model import Word2VecModel
from utils.preprocessing import DefaultPreprocess
from utils.data_helper import build_sorted_vocabulary
from utils.data_helper import build_skip_gram_dataset
from utils.data_helper import Word2VecDataset


class Word2Vec:
    """Provides embeddings from a given text file.

    Attributes
    __________
    word2vec_model : utils.model.Word2VecModel
        PyTorch model you are going to use.

    preprocessor : utils.preprocessing.DefaultPreprocess
         Preprocessing-routine for your particular text.

     Methods
    -------
    fit
        Inside is everything you need to obtain embeddings
    split_data : static
        Is used inside fit
    train_model : static
        Is used inside fit
    test_model : static
        Is used inside fit
    closest_embeddings: static
        Finds the closest words for a given one in the embedding layer
    """

    def __init__(self, word2vec_model=None, preprocessor=None):
        """Choose model and preprocessor you need for your text.

        Parameters
        ----------
        word2vec_model
            If None the word2vec model from model.py will be used.
        preprocessor
            If None the standard preprocessor for Hegel and Kant will be used.
        """

        if preprocessor is None:
            self.preprocessor = DefaultPreprocess()
        else:
            self.preprocessor = preprocessor

        # if word2vec_model is None:
        #     self.word2vec_model = Word2VecModel()
        # else:
        #     self.word2vec_model = word2vec_model

    def fit(self, text: str, eng: bool = True, de: bool = False, model_type: str = 'skipgram',
            remove_stop_words: bool = False, sort_freq: int = 10, window_size: int = 3, n_epochs: int = 5) \
            -> Tuple[Dict[str, int], List[str], np.ndarray]:

        # TODO: instead of using boolean param 'eng' and 'de', make 'lang' = Union['eng': str, 'de': str]

        """Provides embeddings from a raw text file.

        Parameters
        ----------
        text : str
            The whole text (as one string) to be processed and on which the model to be trained.
        eng : bool
            If True then preprocessing for English will be used.
        de  : bool
            If True then preprocessing for German will be used.
        model_type : str
            Skip-Gram ('skipgram') or CBOW ('cbow')
        remove_stop_words : bool
            If True all stop words will be removed.
        sort_freq : int
            Tokens whose frequency of occurrence in the text is more than sort_freq will be added to the vocabulary.
        window_size : int
            Radius of context for a given center-word.
        n_epochs : int
            Number of epochs for the model's training process.

        Returns
        -------
        de_vocab_freq : Dict[str, int]
            Vocabulary sorted by frequency in descending order.
        vocab : List[str]
            Vocabulary as list of words.
            # TODO we already have de_vocab_freq
        embeddings : np.ndarray
            Embedding layer from Word2Vec PyTorch model.
        """

        # ENGLISH TEXT
        if (eng is True) and (de is False):

            eng_raw_text = text

            if remove_stop_words:
                eng_text_as_tokens = self.preprocessor(eng_raw_text, language='english',
                                                       remove_stop_words=True)
                print(f'The number of tokens without stop words is: {len(eng_text_as_tokens)}\n')
            else:
                eng_text_as_tokens = self.preprocessor(eng_raw_text, language='english',
                                                       remove_stop_words=False)
                print(f'The number of tokens with stop words is: {len(eng_text_as_tokens)}\n')

            # PREPARE DATA AND TRAIN THE CHOSEN MODEL

            # TODO: the only fork we need is during the preprocessing.
            #   Remove eng_ and de_ text_as_tokens.
            #   Keep just text_as_tokens etc. since we cannot have both
            #   an English and a German texts simultaneously.
            #   Then just make a fork for skipgram and cbow
            #   Long story short: forking ends after preprocessing part
            #   It's not to be extended over model_type
            #   Result: we don't fork for the the model_type twice as it was
            #   previously for English and German

            if model_type == 'skipgram':
                # build vocabulary
                # eng_vocab_length will be used then in y.append(vocab_len) as id for <UNK>
                eng_vocab_freq, eng_vocab_df, eng_vocab_length = build_sorted_vocabulary(eng_text_as_tokens,
                                                                                         sort_freq)
                # get train data for skip-gram
                X, y, vocab = build_skip_gram_dataset(eng_text_as_tokens, eng_vocab_df, eng_vocab_length,
                                                      window_size)
                # split train data for skip-gram
                X_train, y_train, X_test, y_test = Word2Vec.split_data(X, y)

                print(f'model_type: {model_type}')
                print(f'n_epochs: {n_epochs}\n')

                model = Word2Vec.train_model(vocab, X_train, y_train, model_type='skipgram', n_epochs=n_epochs)

            # TODO: data preparation for CBOW (English)
            # elif model_type == 'cbow':
            # model = self.train_model(vocab, X_train, y_train, model_type='cbow', n_epochs=n_epochs)

            # EVALUATE/TEST THE MODEL
            top_k_accuracy = Word2Vec.test_model(model, X_test, y_test)
            print()
            print(f'top_k_accuracy: {top_k_accuracy:.4f} %')

            # RETURN EMBEDDINGS AS NUMPY ARRAY
            embeddings = model.embeddings.weight.detach().numpy()

            return eng_vocab_freq, vocab, embeddings

        # GERMAN TEXT
        elif (eng is False) and (de is True):

            de_raw_text = text
            if remove_stop_words:
                de_text_as_tokens = self.preprocessor(de_raw_text, language='german',
                                                      remove_stop_words=True)
                print(f'The number of tokens without stop words is: {len(de_text_as_tokens)}\n')
            else:
                de_text_as_tokens = self.preprocessor(de_raw_text, language='german',
                                                      remove_stop_words=False)

                print(f'The number of tokens with stop words is: {len(de_text_as_tokens)}\n')

            # PREPARE DATA AND TRAIN THE MODEL
            if model_type == 'skipgram':
                # build vocabulary
                # de_vocab_length will be used then in y.append(vocab_len) as id for <UNK>
                de_vocab_freq, de_vocab_df, de_vocab_length = build_sorted_vocabulary(de_text_as_tokens,
                                                                                      sort_freq)
                # get train data for skip-gram
                X, y, vocab = build_skip_gram_dataset(de_text_as_tokens, de_vocab_df, de_vocab_length,
                                                      window_size)
                # split train data for skip-gram
                X_train, y_train, X_test, y_test = Word2Vec.split_data(X, y)

                print(f'model_type: {model_type}')
                print(f'n_epochs: {n_epochs}\n')

                model = Word2Vec.train_model(vocab, X_train, y_train, model_type='skipgram', n_epochs=n_epochs)

            # TODO: data preparation for CBOW (Deutsch)
            # elif model_type == 'cbow':
            # model = self.train_model(vocab, X_train, y_train, model_type='cbow', n_epochs=n_epochs)

            # EVALUATE/TEST THE MODEL
            top_k_accuracy = Word2Vec.test_model(model, X_test, y_test)
            print()
            print(f'top_k_accuracy: {top_k_accuracy:.4f} %')

            # RETURN EMBEDDINGS
            embeddings = model.embeddings.weight.detach().numpy()
            return de_vocab_freq, vocab, embeddings

    @staticmethod
    def split_data(X: List[int], y: List[int]) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                        np.ndarray]:
        """
        Parameters
        ----------
        X : List[int]
            Skip-Gram: center words.
            CBOW: radius/context words.
        y : List[int]
            Skip-Gram: radius/contex words.
            CBOW: center words.

        Returns
        -------
        X_train : torch.LongTensor
        y_train : torch.LongTensor
        X_test : torch.LongTensor
        y_test : np.ndarray:
            Since for y_pred in test_model() np.ndarray will be used as well.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.LongTensor(X_test)
        y_test = np.array(y_test)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def train_model(vocab: List[str], X_train: torch.LongTensor, y_train: torch.LongTensor, model_type: str = 'skipgram',
                    batch_size: int = 2**13, n_epochs: int = 5) -> utils.model.Word2VecModel:
        """Trains the model and plots the learning curve.

        Parameters
        ----------
        vocab : List[str]
        X_train : torch.LongTensor
        y_train : torch.LongTensor
        model_type : str
            Skip-Gram ('skipgram') or CBOW ('cbow').
        batch_size : int
        n_epochs : int

        Returns
        -------
        model : utils.model.Word2VecModel
        """

        # TRAIN-LOADER
        # batch_size: 2^13 = 8192, usually 2 to some power
        train_dataset = Word2VecDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # INITIALIZE THE MODEL
        model = Word2VecModel(vocab_vocab_len=len(vocab), model_type=model_type, embd_dim=300, embd_max_norm=1)
        loss_function = nn.CrossEntropyLoss()
        learning_rate = 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # TRAIN MODEL
        model.train()

        all_losses = []

        print('learning process:\n')
        for epoch in range(n_epochs):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                # clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # forward pass to get output/logits
                output_train = model(x_batch)

                # calculate loss: cross entropy loss already has softmax under the hood
                loss = loss_function(output_train, y_batch)

                # getting gradients w.r.t. parameters
                loss.backward()

                # updating parameters: embedding layer and weights of the linear layer
                optimizer.step()

                loss_np = loss.detach().numpy()
                batch_losses.append(loss_np)

            average_loss = float(np.mean(batch_losses))
            all_losses.append(average_loss)
            print(f"epoch #{epoch:<4}: {average_loss:.5f}")

        epochs_np = np.arange(0, n_epochs, 1)
        all_losses_np = np.asarray(all_losses)

        # LEANING CURVE PLOT
        plt.figure(figsize=(20, 10))
        plt.title('Learning Curve')
        plt.plot(epochs_np, all_losses_np)

        return model

    @staticmethod
    def test_model(model: utils.model.Word2VecModel, X_test: torch.LongTensor, y_test: np.ndarray,
                   batch_size: int = 2**13, top_k: int = 9) -> float:
        """
        Parameters
        ----------
        model : utils.model.Word2VecModel
            already trained model to be tested
        X_test : torch.LongTensor
        y_test : np.ndarray
        batch_size : int
        top_k : int

        Returns
        -------
        top_k_accuracy : float
        """

        # TEST-LOADER
        test_dataset = Word2VecDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        model.eval()

        all_outputs = []

        for x_batch, y_batch in test_loader:
            # forward pass to get output/logits
            output_train = model(x_batch)
            # choose top k
            all_outputs.append(torch.topk(torch.nn.functional.softmax(output_train, dim=1), top_k)[1].detach().numpy())

        y_pred = np.row_stack(all_outputs)

        # EVALUATION: Top-k-accuracy
        tp = 0

        for true_id, pred_ids in zip(y_test, y_pred):
            if true_id in pred_ids:
                tp += 1

        top_k_accuracy = (tp / len(y_test)) * 100
        return top_k_accuracy

    @staticmethod
    def closest_embeddings(given_word: str, n_closest: int, vocab: List[str], embeddings: np.ndarray,
                           dist_type: str = 'cosine') -> List[str]:
        """Finds a desired number of the closest words for a given_word.

        Actually, Euclidian or Cosine distances give the same results
        since we used normalization (embd_max_norm = 1) in the embedding layer.

        Parameters
        ----------
        given_word : str
            The word must be in vocabulary.
        n_closest : int
           How many closest words to the given_word should be found.
        vocab : List[str]
        embeddings : np.ndarray
            Embedding layer from the model.
        dist_type : str
            'euclid' or 'cosine'.

        Returns
        -------
        List[str]
            The function returns n_closest words for the given_word.
        """

        if given_word in vocab:
            id_in_vocab = vocab.index(given_word)
        else:
            id_in_vocab = len(vocab)  # = id of <UNK>

        our_word_embedding = embeddings[id_in_vocab]
        word_and_its_distance_tuples = []

        for embedding_id in range(embeddings.shape[0]):
            if embedding_id != id_in_vocab:
                if dist_type == 'euclid':
                    euclid_dist = euclidean(our_word_embedding, embeddings[embedding_id])
                    word_and_its_distance_tuples.append((vocab[embedding_id], euclid_dist))
                elif dist_type == 'cosine':
                    cosine_dist = cosine(our_word_embedding, embeddings[embedding_id])
                    word_and_its_distance_tuples.append((vocab[embedding_id], cosine_dist))

        word_and_its_distance_tuples = sorted(word_and_its_distance_tuples, key=lambda x: x[1])

        return [x[0] for x in word_and_its_distance_tuples[:n_closest]]
