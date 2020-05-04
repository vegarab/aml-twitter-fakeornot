from tfn.models.model import Model
from tfn.preprocess import split_binary_classes
from tfn.feature_extraction.tf_idf import get_tfidf_model

from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity(Model):
    def fit(self, X, y):
        self.x0, self.y0, self.x1, self.y1 = split_binary_classes(X, y)

        self.vectorizer0, self.corpus_matrix0, _ = get_tfidf_model(self.x0)
        self.vectorizer1, self.corpus_matrix1, _ = get_tfidf_model(self.x1)

    def predict(self, X):
        y = []
        for x in X:
            score0 = sum(cosine_similarity(self.vectorizer0.transform([x]), self.corpus_matrix0)[0].tolist())
            score1 = sum(cosine_similarity(self.vectorizer1.transform([x]), self.corpus_matrix1)[0].tolist())

            y.append(self.y0[0] if score0 > score1 else self.y1[0])

        return y


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

    cosine = CosineSimilarity()
    cosine.fit(X_train, y_train)

    y_pred = cosine.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('TF-IDF + cosine-sim accuracy:', round(acc, 4))
    print('TF-IDF + cosine-sim AUC:', round(roc, 4))
    print('TF-IDF + cosine-sim F1:', round(f1, 4))

    if args.export:
        export_results(acc=acc, roc=roc, f1=f1)