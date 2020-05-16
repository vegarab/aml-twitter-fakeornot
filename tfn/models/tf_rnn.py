import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import os


# Remove some tensorflow messages. Set to 1 if you want all outputs (e.g. to 1
# see if a GPU is detected for training)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Read input data from the dataset
data = pd.read_csv('tfn/data/train.csv', header=0)
data = data[['text','target']]

max_features = 10000


def pre_processing(data):
    # Lower case all of the text
    data['text'] = [text.lower() for text in data['text']]

    # Remove anything that it not a letter or a number
    data['text'] = [re.sub('[^a-zA-z0-9\s]', '', text) for text in data['text']]

    # Remove all retweet annotations
    data['text'] = [text.replace('rt', ' ') for text in data['text']]

    tokenizer = Tokenizer(num_words=max_features, split=' ')

    # Train the tokenizer on our dataset
    tokenizer.fit_on_texts(data['text'].values)

    # Encoding words into numbers
    x = tokenizer.texts_to_sequences(data['text'].values)

    # Convert into numpy array
    x = pad_sequences(x)

    return x


def split_train_test(x, data):
    ''' Split the dataset into a test-set and a training-set. Uses a 90-10
        split '''
    # Split out labels from dataset
    y = pd.get_dummies(data['target']).values

    # Split data with a 90-10 split with random seed 42 (this is normally used)
    return train_test_split(x, y, test_size = 0.1, random_state = 42)


def build_model():
    ''' Build a model with 1 LSTM layer using dropout '''
    embed_dim = 100
    lstm_out = 128
    mlp_hidden = 128

    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length = x.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(mlp_hidden, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.2)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics = ['accuracy'])
    print(model.summary())
    return model
        

def visualize(history, epochs):
    ''' Visualize loss during training ''' 
    plt.figure(figsize=(epochs, 8))
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.plot(history['acc'], label='Training accuracy')
    plt.plot(history['val_acc'], label='Validation accuracy')
    plt.legend()
    plt.show()


x = pre_processing(data)
x_train, x_test, y_train, y_test = split_train_test(x, data)

print('Training data shapes:')
print(x_train.shape,y_train.shape)
print('Test data shapes:')
print(x_test.shape,y_test.shape)
print()

# Return a built model, using 1 LSTM layer with dropout
model = build_model()

batch_size = 128
epochs = 5

# Train the model and evaluate on the test set
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=0.2, verbose=1)
score, acc = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)

print('Score: ', (score))
print('Accuracy: :', (acc))

# Visualize the training history
visualize(model.history.history, epochs)
