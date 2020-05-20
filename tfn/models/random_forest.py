from tfn.models.model import Model
from tfn.data_augmentation.augmentation import AugmentWithEmbeddings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skopt.utils import Integer, Categorical
from sklearn.model_selection import cross_val_score


class RandomForest(Model):
    def __init__(self, **params):
        self.clf = RandomForestClassifier(**params)

    def fit(self, X, y, embedding_type='tfidf', glove=None):
        super(RandomForest, self).fit(X, y, embedding_type, glove)
        if self.embedding_type != 'tfidf':
            self.corpus_matrix = np.sum(self.corpus_matrix, axis=1)
        self.clf.fit(self.corpus_matrix, y)

    def predict(self, X):
        super(RandomForest, self).predict(X)
        if self.embedding_type != 'tfidf':
            self.X_transform = np.sum(self.X_transform, axis=1)
        return self.clf.predict(self.X_transform)

    def get_params(self, **kwargs):
        return self.clf.get_params(**kwargs)

    @classmethod
    def get_space(cls):
        return [Categorical(['glove', 'char', 'tfidf'], name='embedding'),
                Categorical([True, False], name='bootstrap'),
                Integer(3, 10, "log-uniform", name='max_depth'),
                Categorical(['auto', 'sqrt'], name='max_features'),
                Integer(1, 4, "log-uniform", name='min_samples_leaf'),
                Integer(2, 5, "log-uniform", name='min_samples_split'),
                Integer(200, 2000, "log-uniform", name='n_estimators')]


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

    rf = RandomForest()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cv = cross_val_score(rf, data.X, data.y, cv=5, n_jobs=-1, scoring="accuracy")
    print(cv)
    print('TF-IDF + Random Forest accuracy:', round(acc, 4))
    print('TF-IDF + Random Forest AUC:', round(roc, 4))
    print('TF-IDF + Random Forest F1:', round(f1, 4))

    if args.export:
        export_results(acc=acc, roc=roc, f1=f1)
