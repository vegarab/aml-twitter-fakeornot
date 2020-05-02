from tfn.models.model import Model
from tfn.preprocess import split_binary_classes
from tfn.feature_extraction.tf_idf import get_tfidf_model
from sklearn.svm import SVC
import numpy as np


class SVM(Model):
    def fit(self, X, y):
        self.vectorizer, self.corpus_matrix, _ = get_tfidf_model(X)

        self.clf = SVC(kernel='rbf')

        print(type(self.corpus_matrix))
        print(self.corpus_matrix.shape)
        print(self.corpus_matrix[0, :])
        self.clf.fit(self.corpus_matrix, y)

    def predict(self, X):
        X_trans = self.vectorizer.transform(X)
        y_pred = self.clf.predict(X_trans)

        return y_pred


if __name__ == '__main__':
    from tfn.preprocess import Dataset
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    data = Dataset('twitter')
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)

    svm = SVM()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    print('TF-IDF + svm accuracy:', round(accuracy_score(y_test, y_pred), 4))
    print('TF-IDF + svm AUC:', round(roc_auc_score(y_test, y_pred), 4))
