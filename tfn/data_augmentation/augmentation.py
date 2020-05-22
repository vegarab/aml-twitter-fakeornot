import pandas as pd
import random
import re
import pickle
import os

from tfn.helper import _get_glove_embeddings, _get_stop_words
from tfn import AUG_PATH
import tqdm


class AugmentWithEmbeddings:
    def __init__(self, emb_size, X, y, replace_pr=0.25, num_copies=4):
        self.glove_emb = _get_glove_embeddings(emb_size=emb_size)
        self.X_aug = []
        self.y_aug = []
        self.stopwords = _get_stop_words()
        pbar = tqdm.tqdm(total=len(X))
        for i in range(len(X)):
            self.X_aug.append([])
            self.y_aug.append([])
            sentence = X[i]
            self.X_aug[i].append(sentence)
            self.y_aug[i].append(y[i])
            for _ in range(num_copies):
                new_sentence = []
                for word in sentence:
                    if word in self.stopwords:
                        new_word = word
                    elif random.random() < replace_pr:
                        new_word = self.replace_word(word)
                        # new_word = word
                    else:
                        new_word = word
                    new_sentence.append(new_word)
                # print(sentence, new_sentence)
                self.X_aug[i].append(new_sentence)
                self.y_aug[i].append(y[i])
            pbar.update(1)
        pbar.close()
        with open(os.path.join(AUG_PATH, "augmented.pkl"), "wb") as f:
            pickle.dump(self.X_aug, f)
            pickle.dump(self.y_aug, f)


    def replace_word(self, word, topn=3, sim_cutoff=0.8):
        word = word.lower()
        org_word = word
        word = re.sub(r"^[^a-zA-Z]*|[^a-zA-Z]*$", "", word)
        try:
            closest = self.glove_emb.similar_by_word(word, topn=topn)
        except KeyError:
            return org_word

        # Keep words with similarity gt cutoff
        closest = [closest[i][0] for i in range(topn) if closest[i][1] > sim_cutoff]

        if closest:
            # Choose word from set of close words
            new_word = random.choice(closest)

            # Substitute the new  word into the formatting of old word e.g. %$..man!! ->  %$..boy!!
            new_word = re.sub(word, new_word, org_word)
            return new_word
        else:
            return org_word

def augment(indices):
    with open(os.path.join(AUG_PATH, "augmented.pkl"), "rb") as f:
        X_aug = pickle.load(f)
        y_aug = pickle.load(f)
        X_out = []
        y_out = []
        for idx in indices:
            X_out += X_aug[idx]
            y_out += y_aug[idx]
    return X_out, y_out

if __name__ == "__main__":
    from tfn.preprocess import Dataset
    data = Dataset('glove')
    AugmentWithEmbeddings(emb_size=25, X=data.X, y=data.y, replace_pr=0.25, num_copies=4)
    # aug = augment([1,2,3])
