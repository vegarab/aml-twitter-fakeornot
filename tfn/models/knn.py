from tfn.models.model import Model
from tfn.feature_extraction.tf_idf import get_tfidf_model
from tfn.feature_extraction.embedding import GloveEmbedding, CharEmbedding

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skopt.utils import Real, Integer, Categorical


class KNN(Model):
    def __init__(self, **params):
        self.clf = KNeighborsClassifier(**params)

    def fit(self, X, y, embedding_type='tfidf', glove=None):
        super(KNN, self).fit(X, y, embedding_type, glove)
        if self.embedding_type != 'tfidf':
            self.corpus_matrix = np.sum(self.corpus_matrix, axis=1)
        self.clf.fit(self.corpus_matrix, y)


    def predict(self, X):
        super(KNN, self).predict(X)
        if self.embedding_type != 'tfidf':
            self.X_transform = np.sum(self.X_transform, axis=1)
        return self.clf.predict(self.X_transform)

    def get_params(self, **kwargs):
        return self.clf.get_params(**kwargs)

    @classmethod
    def get_space(cls):
        return [Categorical(['glove', 'char', 'tfidf'], name='embedding'),
                Integer(1, 30, name='leaf_size'),
                Integer(1, 2, name='p'),
                Integer(1, 10, name='n_neighbors'),
                Categorical(['uniform', 'distance'], name='weights')]


if __name__ == '__main__':
    from tfn.preprocess import Dataset
    from tfn.helper import export_results
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.model_selection import train_test_split
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--export-results", dest="export", action='store_true',
                        help="Exports results to results.csv")
    args = parser.parse_args()

    data = Dataset('twitter')
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)

    knn = KNN()
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('TF-IDF + kNN accuracy:', round(acc, 4))
    print('TF-IDF + kNN AUC:', round(roc, 4))
    print('TF-IDF + kNN F1:', round(f1, 4))

    if args.export:
        export_results(acc=acc, roc=roc, f1=f1)
