from tfn.models.model import Model
from tfn.preprocess import split_binary_classes
from tfn.feature_extraction.tf_idf import get_tfidf_model

from sklearn.neighbors import KNeighborsClassifier

# Class prediction using k-Nearest Neighbors algorithm

class KNN(Model):
    def fit(self, X, y):
        self.vectorizer, self.corpus_matrix, _ = get_tfidf_model(X)

        self.clf = KNeighborsClassifier()
        self.clf.fit(self.corpus_matrix, y)

    def predict(self, X):
        X_emb = self.vectorizer.transform(X)
        y_pred = self.clf.predict(X_emb)
        return y_pred


if __name__ == '__main__':
    from tfn.preprocess import Dataset
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    data = Dataset('twitter')

    knn = KNN()
    knn.fit(data.X_train, data.y_train)

    y_pred = knn.predict(data.X_test)

    print('TF-IDF + kNN accuracy:', round(accuracy_score(data.y_test, y_pred), 4))
    print('TF-IDF + kNN AUC:', round(roc_auc_score(data.y_test, y_pred), 4))

