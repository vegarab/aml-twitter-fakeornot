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

    naive = Naive_Bayes()
    naive.fit(X_train, y_train)
    y_pred = naive.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('TF-IDF + Naive_Bayes accuracy:', round(acc, 4))
    print('TF-IDF + Naive_Bayes AUC:', round(roc, 4))
    print('TF-IDF + Naive_Bayes F1:', round(f1, 4))

    if args.export:
        export_results(acc=acc, roc=roc, f1=f1)