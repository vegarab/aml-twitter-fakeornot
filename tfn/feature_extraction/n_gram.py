from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import defaultdict
import string

from tfn.preprocess import Dataset


def _tokenizer(text):
    ''' Placeholder function for already tokenized inputs '''
    return text


def get_ngram_model(corpus, n_gram=2):
    vectorizer = CountVectorizer(tokenizer=_tokenizer, lowercase=False,
                                 min_df=1, ngram_range=(n_gram, n_gram))  
    corpus_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()

    return vectorizer, corpus_matrix, feature_names


def generate_ngrams(text, n_gram=1):
    ngrams = zip(*[text[i:] for i in range(n_gram)])
    return ngrams


def gram_fig(n_gram=1,target=1,graph=False,name=""):
    df_train = pd.read_csv(_TRAIN_DATA_PATH, dtype={'id': np.int16, 'target': np.int8}, header=0)

    disaster_bigrams = defaultdict(int)

    df_train[:]['text']= Dataset('twitter')._tokenize(df_train[:]['text'],strip_handles=True, strip_rt=True, strip_digits=True)

    '''
    target_index=np.where(df_train[:]['target'] == target)
    #print(target_index)
    '''
    DISASTER_TWEETS = df_train['target'] == target


    for target_self  in df_train[DISASTER_TWEETS]['text']:
        for word in generate_ngrams(target_self, n_gram=n_gram):
            disaster_bigrams[word] += 1

    sort_disaster_unigrams=sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1]

    df_disaster_unigrams = pd.DataFrame(sort_disaster_unigrams)
    df_disaster_unigrams = pd.DataFrame(sort_disaster_unigrams,index=df_disaster_unigrams[:][0])
    df_disaster_unigrams.head()
    #print(df_disaster_unigrams[0:25])
    #print (df_disaster_unigrams[0:10][1])
    #df_disaster_unigrams[2:22].plot.barh(subplots=False,title=name)
    plt.show()

    grams_list = df_disaster_unigrams[:][1].values.tolist()
    grams_name = df_disaster_unigrams[:][0].values.tolist()
    #print(grams_name[0:10])
    ##remove
    bad_chars = [',',"'",'"', ')',"("]
    bad =['\\x89','\\x9d','÷','u','û']
    ii=0
    while(ii<len(grams_name)):
        for i in bad_chars :
            grams_name[ii] = str(grams_name[ii]).replace(i, '')
        ii = ii + 1
    ii=0
    dellete =False
    while(ii<len(grams_name)):
        for i in bad :
            if grams_name[ii].find(i)!=-1:
                grams_name.pop(ii)
                grams_list.pop(ii)
                ii = 0 #reset
                dellete =True
            else :dellete =False

        if dellete ==False:
            ii = ii + 1

    if graph ==True:

        print(grams_name[0:100])
        plt.figure(figsize=(9,25))
        plt.barh(grams_name[0:50][::-1], grams_list[0:50][::-1])
        # Create names on the y-axis
        plt.yticks(grams_name[0:50][::-1], grams_name[0:50][::-1])

        plt.title(name)
        # Show graphic
        plt.show()

    #print(grams_name[0:10])
    pos = 0
    while(grams_list[pos] >=5):
        pos = pos +1
    #print(pos)

    return grams_name




#example()
gettt = gram_fig(n_gram=1,target=1,graph =True,name= "UniGram disaster")
#gettt = gram_fig(n_gram=1,target=0,graph =True,name= "UnGram Nondisaster")
#gettt = gram_fig(n_gram=2,target=1,graph =True,name= "BiGram disaster")
#gettt = gram_fig(n_gram=2,target=0,graph =True,name= "BiGram Nondisaster")
#gettt = gram_fig(n_gram=3,target=1,graph =True,name= "TriGram disaster")
#gettt = gram_fig(n_gram=3,target=0,graph =True,name= "TriGram Nondisaster")
#gettt = gram_fig[0:25]
#print(gettt)
