from tfn.models.model import Model
from tfn.preprocess import split_binary_classes
from tfn.feature_extraction.tf_idf import get_tfidf_model

from sklearn.metrics.pairwise import cosine_similarity


class CosineSimularity(Model):
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
    from sklearn.metrics import accuracy_score

    data = Dataset('twitter')

    cosine = CosineSimularity()
    cosine.fit(data.X, data.y)

    y_pred = cosine.predict(data.X)

    print('TF-IDF + cosine-simularity accuracy:', accuracy_score(data.y, y_pred))


