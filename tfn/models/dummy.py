from tfn.models.model import Model
from tfn.preprocess import split_binary_classes
from tfn.feature_extraction.tf_idf import get_tfidf_model

from collections import Counter

# Baseline naive predictor
# Predicts the class with the most observations for all observations

class Dummy(Model):
    def fit(self, X, y):
        self.dummy_pred = sorted(Counter(y).items(), key=lambda x: x[1], reverse=True)[0][0]

    def predict(self, X):
        y_pred = [self.dummy_pred]*len(X)
        return y_pred

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

    data = Dataset('glove')
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)

    dummy = Dummy()
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('Dummy accuracy:', round(acc, 4))
    print('Dummy AUC:', round(roc, 4))
    print('Dummy F1:', round(f1, 4))

    if args.export:
        export_results(name='Dummy', acc=acc, roc=roc, f1=f1)