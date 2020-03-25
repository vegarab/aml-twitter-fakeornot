
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from pprint import pprint as print
from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath

from tfn.preprocess import Dataset

'''
from gensim.test.utils import common_texts
model = FT_gensim(size=300, window=3, min_count=3)  # instantiate
model.build_vocab(sentences=common_texts)
model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)  # train
'''

'''
from gensim.test.utils import get_tmpfile
fname = get_tmpfile("fasttext2vec.model")
model.save(fname)
model = FT_gensim.load(fname)
'''
def word2vec_model(corpus=trainX,update =False):
    print('trainning')
    data=[]
    for i in corpus:
        data.append(str(i))
    new_data=[]
    i=0
    while(i<len(data)):
        temp = data[i].split()
        new_data.append(temp)
        i= i + 1
    # from gensim.models import FastText  # FIXME: why does Sphinx dislike this import?
    #from gensim.test.utils import common_texts  # some example sentences

    model = FT_gensim(size=300, window=3, min_count=1)  # instantiate
    model.build_vocab(sentences=new_data,update =update )
    model.train(sentences=new_data, total_examples=len(new_data), epochs=10)  # train

    #model.save(fname)
    print('Finish')

    #model = FT_gensim.load(fname)
    return model

def tsne_plot(model,new_vocab,name):
    labels = []
    wordvecs = []

    for word in new_vocab:
        wordvecs.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=3, n_components=2, init='pca', random_state=42)
    coordinates = tsne_model.fit_transform(wordvecs)

    x = []
    y = []
    for value in coordinates:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(8,8))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(2, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title(name)
    plt.show()

def get_word(gram=1,target=1,length =100):
    #load ngran data
    from tfn.feature_extraction.ngram import gram_fig
    vocab = gram_fig(gram,target)

    vocab = vocab[0:length]
    #print(vocab)
    new_vocab =[]
    #seperate word
    i=0
    while(i<len(vocab)):
        temp = vocab[i].split()
        for ii in temp:
            new_vocab.append(ii)
        i= i + 1
    #print(new_vocab)
    #remove duplicate word in list
    newlist = sorted(set(new_vocab), key=lambda x:new_vocab.index(x))
    #print(newlist)
    print('Get '+str(gram)+'-gram '+str(target)+' target')

    return newlist


if __name__ == '__main__':
    trainX , y = Dataset('twitter')._get_training_data_from_csv()

    vocab1_diaster=get_word(gram =1,target =1,length =500)
    vocab1_nondiaster=get_word(gram =1,target =0,length =50)
    vocab2_diaster=get_word(gram =2,target =1,length =50)
    vocab2_nondiaster=get_word(gram =2,target =0,length =50)
    vocab3_diaster=get_word(gram =3,target =1,length =50)
    vocab3_nondiaster=get_word(gram =3,target =0,length =50)


    word2vecmod=word2vec_model(corpus=trainX,update =False)

    tsne_plot(word2vecmod,vocab1_diaster,"UniGram diaster word embedding")
    tsne_plot(word2vecmod,vocab1_nondiaster,"UniGram nondiaster word embedding")
    tsne_plot(word2vecmod,vocab2_diaster,"BiGram diaster word embedding")
    tsne_plot(word2vecmod,vocab2_nondiaster,"BiGram nondiaster word embedding")
    tsne_plot(word2vecmod,vocab3_diaster,"TriGram diaster word embedding")
    tsne_plot(word2vecmod,vocab3_nondiaster,"TriGram nondiaster word embedding")
