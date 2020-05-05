import os

import numpy as np
import pandas as pd
import pickle
import random
import re
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from tfn import TRAIN_FILE

from scipy.spatial import cKDTree

class AugmentWithEmbeddings:
    def __init__(self, replace_pr=0.5):
        self.corpus, self.y = self._get_training_data_from_csv()
        self.idx_map = self._get_idx_mapping()
        self.inv_idx_map = {v: k for k, v in self.idx_map.items()}
        self.glove_emb = self._get_glove_embeddings()

        augmented_data = {'text': [], 'target': []}
        for i in range(len(self.corpus)):
            for _ in range(5):
                sentence = self.corpus[i].split()
                new_sentence = []
                for word in sentence:
                    if random.random() < replace_pr:
                        new_word = self.replace_word(word)
                    else:
                        new_word = word
                    new_sentence.append(new_word)
                new_sentence = " ".join(new_sentence)
                print(sentence, new_sentence)
                augmented_data['text'].append(new_sentence)
                augmented_data['target'].append(self.y[i])
        self.augmented = pd.DataFrame(augmented_data)


    def replace_word(self, word, topn=5):
        org_word = word
        word = word.lower()
        word = re.sub(r'[^\w]', '', word)
        # try:
        closest = self.glove_emb.similar_by_word(word, topn=topn)
        closest = [closest[i][0] for i in range(topn) if closest[i][1] > 0.8]
        try:
            new_word = random.choice(closest)
            new_word = re.sub(word, new_word, org_word)
            return new_word
        except Exception as e:
            print(e)
            return org_word

    def save_data(self):
        save_path = "../data/augmented.csv"
        self.augmented.to_csv(save_path)

    def _get_training_data_from_csv(self):
        df = pd.read_csv(TRAIN_FILE, header=0)
        X = df['text'].to_numpy()
        y = df['target'].to_numpy()

        return X, y

    def _get_idx_mapping(self):
        idx_file = "../misc/idx_map.p"
        with open(idx_file, 'rb') as f:
            idx_map = pickle.load(f)
        return idx_map

    def _get_glove_embeddings(self):
        glove_file = "../misc/glove_25.wv"
        if not os.path.exists(glove_file):
            glove_raw_file = "../misc/glove.twitter.27B.25d.txt"
            glove2word2vec(glove_raw_file, glove_file)
        model = KeyedVectors.load_word2vec_format(glove_file)

        return model


if __name__ == "__main__":
    aug = AugmentWithEmbeddings()