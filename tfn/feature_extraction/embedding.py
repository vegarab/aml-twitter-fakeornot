import numpy as np
import pickle
import os
from pathlib import Path
import warnings


class GloveEmbedding:
    def __init__(self, corpus, type="word", emb_size=50):
        self.corpus = corpus
        misc_dir = Path("../misc/")
        self.type = type
        if self.type == "word":
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
            if self.type == "word":
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

if __name__ == "__main__":
    from tfn.preprocess import Dataset
    ds = Dataset('char')
    emb = GloveEmbedding(ds.X, type='char')

    print(emb.corpus_vectors.shape)
    print(emb.corpus[0])
    print(emb.corpus_vectors[0])