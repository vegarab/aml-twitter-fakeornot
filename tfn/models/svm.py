from tfn.models.model import Model

import numpy as np
from sklearn.svm import SVC
from skopt.utils import Real, Integer, Categorical
from tfn.feature_extraction import embedding

class SVM(Model):
    def __init__(self, **params):
        self.clf = SVC(**params)

    def fit(self, X, y, embedding_type='tfidf', glove=None):
        super(SVM, self).fit(X, y, embedding_type, glove)
        if self.embedding_type != 'tfidf':
            self.corpus_matrix = np.sum(self.corpus_matrix, axis=1)
        self.clf.fit(self.corpus_matrix, y)

    def predict(self, X):
        super(SVM, self).predict(X)
        if self.embedding_type != 'tfidf':
            self.X_transform = np.sum(self.X_transform, axis=1)
        return self.clf.predict(self.X_transform)

    def get_params(self, **kwargs):
        return self.clf.get_params(**kwargs)

    @classmethod
    def get_space(cls):
        return [#Categorical(['tfidf'], name='embedding'),
                # Categorical(['glove', 'char', 'tfidf'], name='embedding'),
                Categorical(['tfidf'], name='embedding'),
                Real(1e-6, 1e+5, "log-uniform", name='C'),
                Real(1e-6, 8, "log-uniform", name='gamma'),
                Integer(1, 8, name='degree'),
                Categorical(['linear', 'poly', 'rbf'], name='kernel')]


if __name__ == '__main__':
    from tfn.preprocess import Dataset
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.model_selection import train_test_split
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", "-p", dest="params", type=dict, default=None,
                        help="Parameter dict for the SVM.")

    args = parser.parse_args()

    data = Dataset('glove')
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)

    glove_init = embedding.GloveEmbedding(emb_size=200)
    svm = SVM()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('TF-IDF + svm accuracy:', round(acc, 4))
    print('TF-IDF + svm AUC:', round(roc, 4))
    print('TF-IDF + svm F1:', round(f1, 4))
