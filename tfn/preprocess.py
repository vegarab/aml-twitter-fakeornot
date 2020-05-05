from collections import Counter

import string
import re

import pandas
import numpy
import spacy

from spellchecker import SpellChecker

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from spacy.lemmatizer import Lemmatizer

from tfn import TRAIN_FILE
from tfn.clean import clean


_TRAIN_DATA_PATH = TRAIN_FILE
_EMOJI_SEQUENCE = ' xx90'

en = spacy.load('en_core_web_sm')
lemmatize = en.Defaults.create_lemmatizer()

START_SPEC_CHARS = re.compile('^[{}]+'.format(re.escape(string.punctuation)))
END_SPEC_CHARS = re.compile('[{}]+$'.format(re.escape(string.punctuation)))


spell = SpellChecker(distance=1)
def check_spelling(tokens, keep_wrong=False):
    if keep_wrong:
        length_original = len(tokens)
        tokens += [
            spell.correction(token) for token in tokens
            if not spell.correction(token) in [
                token for token in tokens
            ]
        ]
        return tokens, len(tokens) - length_original

    elif not keep_wrong:
        corrections = [
            (token, spell.correction(token)) for token in tokens
            if not token == spell.correction(token)
        ]
        for correction in corrections:
            tokens.remove(correction[0])
            tokens.append(correction[1])

        return tokens, len(corrections)


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
                                  strip_digits=True,
                                  strip_hashtags=False,
                 test_size=0.3):

        # Get raw data
        self.corpus, self.y = self._get_training_data_from_csv()
        self.y = self.y.tolist()

        if tokenizer == 'twitter':
            self.X = self._tokenize(self.corpus, 
                                    strip_handles, 
                                    strip_rt, 
                                    strip_digits,
                                    strip_hashtags)
        elif tokenizer == 'lemmatize':
            self.X = self._tokenize_with_lemma(self.corpus, 
                                               strip_handles, 
                                               strip_rt,
                                               strip_digits)
        elif tokenizer == 'glove':
            self.X = self._tokenize_glove(self.corpus)
        elif tokenizer == 'char':
            self.X = self._tokenize_character(self.corpus)
        else:
            raise AttributeError("This functions only accepts 'twitter', 'lemmatize', 'glove' and "
                               + "'char' as possible tokenizers")

        # Might be betteer to save splitting for outside the dataset so as to preserve the order of entries?
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)

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
            # Add special sequence for emojis (??). Needs to be done before any
            # punctuation removal or tokenization
            doc = doc.replace('??', _EMOJI_SEQUENCE)

            # Applies cleaning from clean.py
            doc = clean(doc)

            # Replace hashtag with <hashtag> token as is encoded in GLoVe
            doc = doc.replace('#', '<hashtag> ')

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

    def _tokenize(self, corpus, strip_handles=True, strip_rt=True, 
                  strip_digits=True, strip_hashtags=False):
        ''' Tokenize corpus using NLTK's TwitterTokenizer '''

        tokenizer = TweetTokenizer(strip_handles=strip_handles, reduce_len=True)
        stop_words = _get_stop_words(strip_handles, strip_rt)

        output = []
        for doc in corpus:
            # Add special sequence for emojis (??). Needs to be done before any
            # punctuation removal or tokenization
            doc = doc.replace('??', _EMOJI_SEQUENCE)
            
            # Applies cleaning from clean.py
            doc = clean(doc)
            
            # Tokenize
            tokens = [word.lower() for word in tokenizer.tokenize(doc)]

            # Remove tokens which contain any number.
            if strip_digits:
                tokens = [token for token in tokens if not _has_digits(token)]

            # Remove tokens without text.
            tokens = [token for token in tokens if bool(token.strip())]

            # Remove punctuation from start of tokens.
            if strip_hashtags:
                tokens = [re.sub(START_SPEC_CHARS, '', token) for token in tokens]

            # Remove punctuation from end of tokens.
            tokens = [re.sub(END_SPEC_CHARS, '', token) for token in tokens]

            # Filters out frequent special characters
            tokens = [token for token in tokens if token not in string.punctuation+'…’']

            # Remove stopwords
            tokens = [token for token in tokens if token not in stop_words]

            output.append(tokens)

        return output

    def _tokenize_glove(self, corpus):
        ''' Tokenize corpus using NLTK's TwitterTokenizer '''

        tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
        stop_words = _get_stop_words(strip_handles=False, strip_rt=False)

        output = []
        for doc in corpus:
            # Add special sequence for emojis (??). Needs to be done before any
            # punctuation removal or tokenization
            doc = doc.replace('??', _EMOJI_SEQUENCE)

            # Applies cleaning from clean.py
            doc = clean(doc)

            # Replace tokens as encoded in GLoVe
            doc = re.sub(r'#\w+', '<hashtag>', doc)
            doc = re.sub(r'@\w+', '<user> ', doc)
            doc = re.sub(r'https?://t\.co/[\w\d]{10}', '<url>', doc)
            doc = re.sub(r'(\d{1,3},?)+', '<number>', doc)
            doc = re.sub(r'\.{2,}', "…", doc)
            doc = doc.replace("'s", " 's")
            doc = re.sub(r":\)", '<smile>', doc)
            doc = re.sub(r"\(:", '<smile>', doc)
            doc = re.sub(r":\(", '<sadface>', doc)
            doc = re.sub(r"\):", '<sadface>', doc)

            doc = doc.replace("-", " ")

            # Tokenize
            tokens = [word.lower() for word in tokenizer.tokenize(doc)]

            # Remove tokens without text.
            tokens = [token for token in tokens if bool(token.strip())]

            # Remove punctuation from start of tokens.
            # if strip_hashtags:
            #     tokens = [re.sub(START_SPEC_CHARS, '', token) for token in tokens]
            #
            # # Remove punctuation from end of tokens.
            # tokens = [re.sub(END_SPEC_CHARS, '', token) for token in tokens]

            # Filters out frequent special characters
            tokens = [token for token in tokens if token not in string.punctuation + '…’']

            # Remove stopwords
            tokens = [token for token in tokens if token not in stop_words]

            output.append(tokens)

        return output

    def _tokenize_character(self, corpus):
        output = []
        char_set = string.ascii_letters + string.digits + string.punctuation + " "
        char_set = char_set.replace(">", "")
        char_set = char_set.replace("<", "")
        for doc in corpus:
            doc = doc.replace(" ", "_")
            tokens = [x for x in doc if x in char_set]
            output.append(tokens)
        return output

if __name__ == '__main__':
   ds = Dataset('glove')
