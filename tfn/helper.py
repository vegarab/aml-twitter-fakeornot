import __main__
from datetime import datetime
import csv
import os
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from tfn import TRAIN_FILE

def export_results(acc, roc, f1):
    results_file = 'data/results.csv'
    model = os.path.basename(__main__.__file__)
    dt = datetime.now()
    fields = [model, dt, acc, roc, f1]
    with open(results_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def _get_glove_embeddings(emb_size=25):
    glove_file = "../misc/glove_%s.wv" % emb_size
    if not os.path.exists(glove_file):
        glove_raw_file = "../misc/glove.twitter.27B.%sd.txt" % emb_size
        glove2word2vec(glove_raw_file, glove_file)
    model = KeyedVectors.load_word2vec_format(glove_file)

    return model


def _get_training_data_from_csv():
    df = pd.read_csv(TRAIN_FILE, header=0)
    X = df['text'].to_numpy()
    y = df['target'].to_numpy()

    return X, y
