import pandas as pd
import random
import re

from tfn.helper import _get_glove_embeddings


class AugmentWithEmbeddings:
    def __init__(self, X, y, replace_pr=0.5):
        self.glove_emb = _get_glove_embeddings()
        self.X_aug = []
        self.y_aug = []
        for i in range(len(X)):
            sentence = X[i]
            self.X_aug.append(sentence)
            self.y_aug.append(y[i])
            for _ in range(5):
                new_sentence = []
                for word in sentence:
                    if random.random() < replace_pr:
                        new_word = self.replace_word(word)
                    else:
                        new_word = word
                    new_sentence.append(new_word)
                print(sentence, new_sentence)
                self.X_aug.append(new_sentence)
                self.y_aug.append(y[i])

    def replace_word(self, word, topn=5):
        word = word.lower()
        org_word = word
        word = re.sub(r"^[^a-zA-Z]*|[^a-zA-Z]*$", "", word)
        try:
            closest = self.glove_emb.similar_by_word(word, topn=topn)
        except KeyError:
            return org_word
        closest = [closest[i][0] for i in range(topn) if closest[i][1] > 0.8]

        if closest:
            new_word = random.choice(closest)
            new_word = re.sub(word, new_word, org_word)
            return new_word
        else:
            return org_word


if __name__ == "__main__":
    pass
