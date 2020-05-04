from tfn.models.model import Model
from tfn.preprocess import split_binary_classes
from tfn.feature_extraction.tf_idf import get_tfidf_model
from sklearn import model_selection, naive_bayes

class Naive_Bayes(Model):
    def fit(self, X, y):

        self.vectorizer, self.corpus_matrix, _ = get_tfidf_model(X)

        self.clf = naive_bayes.MultinomialNB()
        self.clf.fit(self.corpus_matrix, y)
    
    def predict(self, X):
        X_trans = self.vectorizer.transform(X)
        y_pred = self.clf.predict(X_trans)

        return y_pred


if __name__ == '__main__':
    from tfn.preprocess import Dataset
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.model_selection import train_test_split

    data = Dataset('twitter')
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y)

    naive = Naive_Bayes()
    naive.fit(X_train, y_train)
    y_pred = naive.predict(X_test)

    print('TF-IDF + Naive_Bayes accuracy:', round(accuracy_score(y_test, y_pred), 4))
    print('TF-IDF + Naive_Bayes AUC:', round(roc_auc_score(y_test, y_pred), 4))
    print('TF-IDF + Naive_Bayes F1:', round(f1_score(y_test, y_pred), 4))