from tfn.feature_extraction import embedding, tf_idf


class Model:
    def fit(self, X, y, embedding_type, glove):
        self.glove = glove
        self.embedding_type = embedding_type
        if self.embedding_type == 'glove':
            self.corpus_matrix = self.glove.corpus_vectors(X, show_errors=False)
        elif self.embedding_type == 'char':
            self.corpus_matrix = embedding.CharEmbedding(X).corpus_vectors
        else:
            self.vectorizer, corpus_matrix, _ = tf_idf.get_tfidf_model(X)
            self.corpus_matrix = corpus_matrix

    def predict(self, X):
        if self.embedding_type == 'glove':
            self.X_transform = self.glove.corpus_vectors(X, show_errors=False)
        elif self.embedding_type == 'char':
            self.X_transform = embedding.CharEmbedding(X).corpus_vectors
        else:
            self.X_transform = self.vectorizer.transform(X)
