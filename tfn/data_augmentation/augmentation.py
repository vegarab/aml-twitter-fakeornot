import pandas as pd
import random
import re

from tfn.helper import _get_glove_embeddings, _get_training_data_from_csv


class AugmentWithEmbeddings:
    def __init__(self, replace_pr=0.5):
        self.corpus, self.y = _get_training_data_from_csv()
        self.glove_emb = _get_glove_embeddings()

        augmented_data = {'text': [], 'target': []}
        for i in range(len(self.corpus)):
            for _ in range(5):
                sentence = self.corpus[i]
                sentence_spl = sentence.split()
                new_sentence = []
                for word in sentence_spl:
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

    def save_data(self):
        save_path = "../data/augmented.csv"
        self.augmented.to_csv(save_path)


if __name__ == "__main__":
    aug = AugmentWithEmbeddings()
