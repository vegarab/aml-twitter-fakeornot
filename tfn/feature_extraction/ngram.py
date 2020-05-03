from sklearn.feature_extraction.text import CountVectorizer


def _tokenizer(text):
    ''' Placeholder function for already tokenized inputs '''
    return text


def get_ngram_model(corpus, n_gram=2):
    vectorizer = CountVectorizer(tokenizer=_tokenizer, lowercase=False,
                                 min_df=1, ngram_range=(n_gram, n_gram))  
    corpus_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()

    return vectorizer, corpus_matrix, feature_names
