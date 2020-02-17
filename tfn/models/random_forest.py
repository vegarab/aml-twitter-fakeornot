from tfn.models.model import Model
from tfn.feature_extraction.tf_idf import get_tfidf_model

from sklearn.ensemble import RandomForestClassifier

class RandomForest(Model):
    def fit(self, X, y):
        self.vectorizer, self.X_vectorized, _ = get_tfidf_model(X)

        self.clf = RandomForestClassifier()
        self.clf.fit(self.X_vectorized, y)

    def predict(self, X):
        X_emb = self.vectorizer.transform(X)
        y_pred = self.clf.predict(X_emb)
        return y_pred

if __name__ == '__main__':
    from tfn.preprocess import Dataset
    from sklearn.metrics import accuracy_score, roc_auc_score

    data = Dataset('twitter')

    rf = RandomForest()
    rf.fit(data.X_train, data.y_train)

    y_pred = rf.predict(data.X_test)

    print('TF-IDF + Random Forest accuracy:', round(accuracy_score(data.y_test, y_pred), 4))
    print('TF-IDF + Random Forest AUC:', round(roc_auc_score(data.y_test, y_pred), 4))