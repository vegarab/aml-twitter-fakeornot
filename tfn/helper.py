from datetime import datetime
import csv
import os
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from tfn import TRAIN_FILE, GLOVE_FILE, GLOVE_WV_FILE, RESULTS_FILE


def export_results(name, acc, roc, f1):
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'a') as f:
            headers = ["Model", "Date", "Accuracy", "ROC-AUC", "F1-Score"]
            writer = csv.writer(f)
            writer.writerow(headers)
    dt = datetime.now()
    fields = [name, dt, acc, roc, f1]
    with open(RESULTS_FILE, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def _get_glove_embeddings(emb_size=25):
    glove_file = GLOVE_WV_FILE % emb_size
    if not os.path.exists(glove_file):
        glove_raw_file = GLOVE_FILE % emb_size
        glove2word2vec(glove_raw_file, glove_file)
    model = KeyedVectors.load_word2vec_format(glove_file)

    return model


def _get_training_data_from_csv():
    df = pd.read_csv(TRAIN_FILE, header=0)
    X = df['text'].to_numpy()
    y = df['target'].to_numpy()

    return X, y
