from sklearn.feature_extraction.text import TfidfVectorizer


def _tokenizer(text):
    ''' Placeholder function for already tokenized inputs '''
    return text


def get_tfidf_model(corpus):
    vectorizer = TfidfVectorizer(tokenizer=_tokenizer, sublinear_tf=True, lowercase=False)
    corpus_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()

    return vectorizer, corpus_matrix, feature_names
