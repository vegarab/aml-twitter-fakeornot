import pandas as pd
import random
import re

from tfn.helper import _get_glove_embeddings, _get_stop_words


class AugmentWithEmbeddings:
    def __init__(self):
        self.glove_emb = _get_glove_embeddings()
        self.X_aug = []
        self.y_aug = []
        self.stopwords = _get_stop_words()

    def augment(self, X, y, replace_pr=0.25, num_copies=4):
        for i in range(len(X)):
            if i % 100 == 0:
                print("Augmentation %s%% complete..." % (100*i // len(X)))
            sentence = X[i]
            self.X_aug.append(sentence)
            self.y_aug.append(y[i])
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
                self.X_aug.append(new_sentence)
                self.y_aug.append(y[i])
        return X_aug, y_aug

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


if __name__ == "__main__":
    pass
