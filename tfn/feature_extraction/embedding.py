import numpy as np
import pickle
import os
from pathlib import Path
import warnings
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

from tfn.helper import _get_glove_embeddings


# Uses GLoVe word embeddings trained by researchers at Stanford on Twitter data
class GloveEmbedding:
    def __init__(self, corpus, type="glove", emb_size=50):
        self.corpus = corpus
        self.type = type
        if emb_size not in [25, 50, 100, 200]:
            raise ValueError("Embedding size must be 25, 50, 100 or 200.")
        self.emb_size = emb_size
        self.wv = _get_glove_embeddings(emb_size=self.emb_size)

    @property
    def corpus_vectors(self):
        max_len = len(max(self.corpus, key=len))
        output = np.zeros(shape=(len(self.corpus), max_len, self.emb_size))
        for i, doc_word_list in enumerate(self.corpus):
            for j, word in enumerate(doc_word_list):
                try:
                    output[i, j, :] = self.wv.get_vector(word)
                except KeyError:
                    print('Embedding not found for word "%s".' % word)
        return output


# Creates character embeddings using large Twitter dataset and gensim Word2Vec model
class CharEmbedding:
    def __init__(self, X, train=False, training_path=None):
        model_path = "../misc/character_model.wv"
        if train:
            self.train(training_path, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError("Character embedding model not found. Please run with train=True to create model.")
        self.X_enc = self.encode(X, model_path)

    # Train character embeddings using external Twitter dataset
    def train(self, training_path, model_path, epochs=100):
        with open(training_path, 'r') as f:
            sentences = [list(x) for x in f.readlines()]
        model = Word2Vec(sentences, iter=epochs, compute_loss=True, callbacks=[self.Callback()])
        model.wv.save(model_path)

    # Encode our data using above embeddings
    def encode(self, X, model_path):
        wv = KeyedVectors.load(model_path)
        max_len = len(max(X, key=len))
        encoded_matrix = np.zeros(shape=(len(X), max_len, wv.vector_size))  # Shape: n x num_chars x vec_dim
        for i in range(len(X)):
            for j in range(len(X[i])):
                char_enc = wv[X[i][j]]
                encoded_matrix[i, j] = char_enc
        return encoded_matrix

    class Callback(CallbackAny2Vec):
        """Callback to print loss after each epoch."""

        def __init__(self):
            self.epoch = 0
            self.loss_previous_step = 0

        def on_epoch_end(self, model):
            loss = model.get_latest_training_loss()
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
            self.epoch += 1
            self.loss_previous_step = loss


if __name__ == "__main__":
    from tfn.preprocess import Dataset

    # Test GloveEmbedding
    type = 'glove'
    ds = Dataset(type)
    emb = GloveEmbedding(ds.X, type=type)

    print(emb.corpus_vectors.shape)
    print(emb.corpus[0])
    print(emb.corpus_vectors[0])

    # Test OneHotCharEmbedding
    # ds = Dataset('char')
    # emb = CharEmbedding(ds.X, train=True, training_path="../data/training.1600000.processed.noemoticon.csv")
