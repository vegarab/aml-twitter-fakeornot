from collections import Counter

import string
import re

import pandas
import numpy
import spacy

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from spacy.lemmatizer import Lemmatizer


_TRAIN_DATA_PATH = 'tfn/data/train.csv'

en = spacy.load('en_core_web_sm')
lemmatize = en.Defaults.create_lemmatizer()

START_SPEC_CHARS = re.compile('^[{}]+'.format(re.escape(string.punctuation)))
END_SPEC_CHARS = re.compile('[{}]+$'.format(re.escape(string.punctuation)))


# TODO: This function can probably be waaaay neater than this mess
def split_binary_classes(X, y):
    ''' Split the dataset into groups of the two classes '''
    x0 = []
    x1 = []
    y0 = []
    y1 = []

    for i,x in enumerate(X):
        if y[i] == 0:
            x0.append(x)
            y0.append(y[i])
        else:
            x1.append(x)
            y1.append(y[i])


    return x0, y0, x1, y1

def _get_stop_words(strip_handles, strip_rt):
    ''' Returns stopwords '''
    stop_words = (stopwords.words('english'))
    if strip_rt: stop_words += ['rt']
    # TODO: if strip_handles
    return set(stop_words)


def _has_digits(token):
    ''' Returns true if the given string contains any digits '''
    return any(char.isdigit() for char in token)


class Dataset():
    def __init__(self, tokenizer, strip_handles=True, 
                                  strip_rt=True, 
                                  strip_digits=True):
        # Get raw data
        self.corpus, self.y = self._get_training_data_from_csv()
        self.y = self.y.tolist()

        if tokenizer == 'twitter':
            self.X = self._tokenize(self.corpus, 
                                    strip_handles, 
                                    strip_rt, 
                                    strip_digits)
        elif tokenizer == 'lemmatize':
            self.X = self._tokenize_with_lemma(self.corpus, 
                                               strip_handles, 
                                               strip_rt,
                                               strip_digits)
        else:
            raise AttributeError("This functions only accepts 'twitter' and "
                               + "'lemmatize' as possible tokenizers")

        
    def _get_training_data_from_csv(self):
        df = pandas.read_csv(_TRAIN_DATA_PATH, header=0)
        X = df['text'].to_numpy()
        y = df['target'].to_numpy()

        return X, y


    def _tokenize_with_lemma(self, corpus, strip_handles=True, strip_rt=True, strip_digits=True):
        ''' Tokenize and lemmatize using Spacy '''
        
        stop_words = _get_stop_words(strip_handles, strip_rt)

        output = []
        for doc in corpus:
            # Tokenize the document.
            tokens = [lemmatize(token.text, token.pos_)[0].lower() for token in en(doc)]

            # Remove punctuation tokens.
            tokens = [token for token in tokens if token not in string.punctuation+'…’']

            # Remove tokens wich contain any number.
            if strip_digits:
                tokens = [token for token in tokens if not _has_digits(token)]

            # Remove tokens without text.
            tokens = [token for token in tokens if bool(token.strip())]

            # Remove punctuation from start of tokens.
            tokens = [re.sub(START_SPEC_CHARS, '', token) for token in tokens]

            # Remove punctuation from end of tokens.
            tokens = [re.sub(END_SPEC_CHARS, '', token) for token in tokens]

            # Remove stopwords from the tokens
            tokens = [token for token in tokens if token not in stop_words]

            output.append(tokens)

        return output

    def _tokenize(self, corpus, strip_handles=True, strip_rt=True, strip_digits=True):
        ''' Tokenize corpus using NLTK's TwitterTokenizer '''

        tokenizer = TweetTokenizer(strip_handles=strip_handles, reduce_len=True)
        stop_words = _get_stop_words(strip_handles, strip_rt)

        output = []
        for doc in corpus:
            # Tokenize
            tokens = [word.lower() for word in tokenizer.tokenize(doc)]

            # Remove tokens which contain any number.
            if strip_digits:
                tokens = [token for token in tokens if not _has_digits(token)]

            # Remove tokens without text.
            tokens = [token for token in tokens if bool(token.strip())]

            # Remove punctuation from start of tokens.
            tokens = [re.sub(START_SPEC_CHARS, '', token) for token in tokens]

            # Remove punctuation from end of tokens.
            tokens = [re.sub(END_SPEC_CHARS, '', token) for token in tokens]

            # Filters out frequent special characters
            tokens = [token for token in tokens if token not in string.punctuation+'…’']

            # Remove stopwords
            tokens = [token for token in tokens if token not in stop_words]
            
            output.append(tokens)

        return output


if __name__ == '__main__':
   ds = Dataset('twitter')
