import numpy as np
import pickle
import os
from pathlib import Path
import warnings
import string
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec


class GloveEmbedding:
    def __init__(self, corpus, type="glove", emb_size=50):
        self.corpus = corpus
        misc_dir = Path("../misc/")
        self.type = type
        if self.type == "glove":
            if emb_size not in [25, 50, 100, 200]:
                raise ValueError("Embedding size must be 25, 50, 100 or 200.")
            self.emb_size = emb_size
            self.glove_file = misc_dir / ("glove.twitter.27B.%sd.txt" % self.emb_size)
            self.vec_file = misc_dir / ("glove.%s.npy" % self.emb_size)
            self.idx_file = misc_dir / "idx_map.p"
        elif self.type == "char":
            if emb_size:
                warnings.warn('Embedding size of %s was used but embedding size is an irrelevent argument for type "char".' % emb_size)
            self.emb_size = 300
            self.glove_file = misc_dir / "glove.840B.300d-char.txt"
            self.vec_file = misc_dir / "glove.char.npy"
            self.idx_file = misc_dir / "idx_map_char.p"
        else:
            raise ValueError()


    @property
    def corpus_vectors(self):
        vector_mapping, word_idx_mapping = self.mapping()
        max_len = len(max(self.corpus, key=len))
        output = np.zeros(shape=(len(self.corpus), max_len, self.emb_size))
        for i, doc_word_list in enumerate(self.corpus):
            for j, word in enumerate(doc_word_list):

                # This will spit out errors on hashtags and @s
                # Also errors for some words that are misspelled
                #TODO: change preprocess.py to accomodate <hashtag> and <url> embeddings.
                try:
                    word_idx = word_idx_mapping[word]
                    output[i, j, :] = vector_mapping[word_idx]
                except KeyError:
                    print('Embedding not found for word "%s".'%word)
        return output

    def mapping(self):
        if os.path.exists(self.vec_file) and os.path.exists(self.idx_file):
            return self.mapping_from_save()
        elif os.path.exists(self.glove_file):
            return self.mapping_from_file()
        else:
            if self.type == "glove":
                raise FileNotFoundError("Mapping files not found. Please run 'get_glove_embeddings.ipynb' from notebooks.")
            else:
                raise FileNotFoundError(
                    "Mapping files not found. Download character embeddings from https://github.com/minimaxir/char-embeddings and place in tfn/misc.")
    def mapping_from_file(self):
        def _file_len(f):
            return len(f.readlines())

        with open(self.glove_file, 'rb') as f:
            line_gen = f.readlines()
            num_words = len(line_gen)
            vector_mapping = np.zeros(shape=(num_words, self.emb_size))
            word_idx_mapping = {}

            for i, line in enumerate(line_gen):
                if i % 10000 == 0:
                    complete_prop = "%s%%"%int(100* i / num_words)
                    print(complete_prop)
                # print(line, '\n')
                line = line.decode().split()
                word = line[0]
                vector = np.array(line[1:]).astype(np.float)
                word_idx_mapping[word] = i
                try:
                    vector_mapping[i] = vector
                except ValueError:
                    print("Line %s resulted in an error. Investigate." % line)

        # Saves vector and index mapping files for future use
        with open(self.idx_file, 'wb') as f:
            pickle.dump(word_idx_mapping, f)
        np.save(self.vec_file, vector_mapping)

        return vector_mapping, word_idx_mapping

    def mapping_from_save(self):
        vector_mapping = np.load(self.vec_file)
        with open(self.idx_file, 'rb') as f:
            word_idx_mapping = pickle.load(f)
        return vector_mapping, word_idx_mapping


class CharEmbedding:
    def __init__(self, X, train=False, training_path=None):
        model_path = "../misc/character_model.wv"
        if train:
            self.train(training_path, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError("Character embedding model not found. Please run with train=True to create model.")
        self.X_enc = self.encode(X, model_path)

    # Train character embeddings using external Twitter dataset
    def train(self, training_path, model_path):
        with open(training_path, 'r') as f:
            sentences = [list(x) for x in f.readlines()]
        model = Word2Vec(sentences, compute_loss=True, callbacks=[self.callback()])
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

    class callback(CallbackAny2Vec):
        '''Callback to print loss after each epoch.'''

        def __init__(self):
            self.epoch = 0

        def on_epoch_end(self, model):
            loss = model.get_latest_training_loss()
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
            self.epoch += 1


if __name__ == "__main__":
    from tfn.preprocess import Dataset

    # Test GloveEmbedding
    # type = 'glove'
    # ds = Dataset(type)
    # emb = GloveEmbedding(ds.X, type=type)
    #
    # print(emb.corpus_vectors.shape)
    # print(emb.corpus[0])
    # print(emb.corpus_vectors[0])

    # Test OneHotCharEmbedding
    ds = Dataset('char')
    emb = CharEmbedding(ds.X, train=True, training_path="../data/training.1600000.processed.noemoticon.csv")
