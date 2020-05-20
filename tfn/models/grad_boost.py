from tfn.models.model import Model

import xgboost as xgb
import numpy as np

from skopt.utils import Categorical, Real, Integer


class GradientBoost(Model):
    def __init__(self, **params):
        self.params = {'objective': 'binary:logistic',
                       'eval_metric': 'rmse'}
        self.params.update(params)
        self.clf = None

    def fit(self, X, y, embedding_type='tfidf', glove=None):
        super(GradientBoost, self).fit(X, y, embedding_type, glove)
        if self.embedding_type != 'tfidf':
            self.corpus_matrix = np.sum(self.corpus_matrix, axis=1)
        epochs = 100
        D_train = xgb.DMatrix(self.corpus_matrix, label=y)
        self.clf = xgb.train(self.params, D_train, epochs)

    def predict(self, X):
        super(GradientBoost, self).predict(X)
        if self.embedding_type != 'tfidf':
            self.X_transform = np.sum(self.X_transform, axis=1)
        D_test = xgb.DMatrix(self.X_transform)
        y_pred = self.clf.predict(D_test)

        # Convert to binary output
        return (y_pred > 0.5).astype(int)

    def get_params(self, **kwargs):
        return self.params

    @classmethod
    def get_space(cls):
        return [Categorical(['glove', 'char', 'tfidf'], name='embedding'),
                Real(1e-3, 0.3, 'log-uniform', name='eta'),
                Real(1e-3, 0.3, 'log-uniform', name='min_child_weight'),
                Integer(3, 10, name='max_depth'),
                Real(1e-6, 1e+6, 'log-uniform', name='gamma')]


if __name__ == "__main__":
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

    grad_boost = GradientBoost()
    grad_boost.fit(X_train, y_train)
    y_pred = grad_boost.predict(X_test)

    print(y_pred)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('TF-IDF + xgb accuracy:', round(acc, 4))
    print('TF-IDF + xgb AUC:', round(roc, 4))
    print('TF-IDF + xgb F1:', round(f1, 4))

    if args.export:
        export_results(acc=acc, roc=roc, f1=f1)
