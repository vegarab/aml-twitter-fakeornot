
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from tfn.models.model import Model
from tfn.feature_extraction.tf_idf import get_tfidf_model
from tfn.feature_extraction.embedding import  GloveEmbedding, CharEmbedding


LABELS = [ 0, 1]
CENTERS = [(-3, -3), (3, 3), (3, -3), (-3, 3)]

class MLP(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size=100, output_size=1,
                 num_hidden_layers=1, hidden_activation=nn.Sigmoid):
        """Initialize weights.

        Args:
            input_size (int): size of the input
            hidden_size (int): size of the hidden layers
            output_size (int): size of the output
            num_hidden_layers (int): number of hidden layers
            hidden_activation (torch.nn.*): the activation class
        """
        super(MLP, self).__init__()
        self.module_list = nn.ModuleList()

        interim_input_size = input_size
        interim_output_size = hidden_size

        for _ in range(num_hidden_layers):
            self.module_list.append(nn.Linear(interim_input_size, interim_output_size))
            self.module_list.append(hidden_activation())
            interim_input_size = interim_output_size

        self.fc_final = nn.Linear(interim_input_size, output_size)

        self.last_forward_cache = []

    def forward(self, x, apply_softmax=False):
        """The forward pass of the MLP

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        self.last_forward_cache = []
        self.last_forward_cache.append(x.to("cpu").numpy())

        for module in self.module_list:
            #x=torch.double(x)

            print(x.type)

            x = module(x)
            self.last_forward_cache.append(x.to("cpu").data.numpy())

        output = self.fc_final(x)
        self.last_forward_cache.append(output.to("cpu").data.numpy())

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output
    def fit(self, X, y, input_size=5,output_size=len(set(LABELS)), num_hidden_layers=1,hidden_size=300):
       pass

def get_toy_data(batch_size):
    assert len(CENTERS) == len(LABELS), 'centers should have equal number labels'

    x_data = []
    y_targets = np.zeros(batch_size)
    n_centers = len(CENTERS)

    for batch_i in range(batch_size):
        center_idx = np.random.randint(0, n_centers)
        x_data.append(np.random.normal(loc=CENTERS[center_idx]))
        y_targets[batch_i] = LABELS[center_idx]

    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.int64)

def visualize_results(perceptron, x_data, y_truth, n_samples=1000, ax=None, epoch=None,
                      title='', levels=[0.3, 0.4, 0.5], linestyles=['--', '-', '--']):
    _, y_pred = perceptron(x_data, apply_softmax=True).max(dim=1)
    y_pred = y_pred.data.numpy()

    x_data = x_data.data.numpy()
    y_truth = y_truth.data.numpy()


    n_classes = len(set(LABELS))

    all_x = [[] for _ in range(n_classes)]
    all_colors = [[] for _ in range(n_classes)]

    colors = ['black', 'white']
    markers = ['o', '*']

    for x_i, y_pred_i, y_true_i in zip(x_data, y_pred, y_truth):
        all_x[y_true_i].append(x_i)
        if y_pred_i == y_true_i:
            all_colors[y_true_i].append("white")
        else:
            all_colors[y_true_i].append("black")
        #all_colors[y_true_i].append(colors[y_pred_i])

    all_x = [np.stack(x_list) for x_list in all_x]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,10))

    for x_list, color_list, marker in zip(all_x, all_colors, markers):
        ax.scatter(x_list[:, 0], x_list[:, 1], edgecolor="black", marker=marker, facecolor=color_list, s=100)


    xlim = (min([x_list[:,0].min() for x_list in all_x]),
            max([x_list[:,0].max() for x_list in all_x]))

    ylim = (min([x_list[:,1].min() for x_list in all_x]),
            max([x_list[:,1].max() for x_list in all_x]))

    # hyperplane

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    for i in range(n_classes):
        Z = perceptron(torch.tensor(xy, dtype=torch.float32),
                       apply_softmax=True)
        Z  = Z[:, i].data.numpy().reshape(XX.shape)
        ax.contour(XX, YY, Z, colors=colors[i], levels=levels, linestyles=linestyles)

    # plotting niceties

    plt.suptitle(title)

    if epoch is not None:
        plt.text(xlim[0], ylim[1], "Epoch = {}".format(str(epoch)))
if __name__ == '__main__':

    from tfn.preprocess import Dataset
    from tfn.helper import export_results
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.model_selection import train_test_split
    import argparse

    from tfn.feature_extraction.word2vec import word2vec_model
    from tfn import TRAIN_FILE
    import os

    from gensim.models import Word2Vec,KeyedVectors
    from torch.utils.data import DataLoader, TensorDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--export-results", dest="export", action='store_true',
                        help="Exports results to results.csv")
    args = parser.parse_args()

    #model.wv.save(model_path)

    #emb = CharEmbedding(data.X,train=True ,training_path = TRAIN_FILE

    data = Dataset('glove')
    model= word2vec_model(data.X)
    with open(TRAIN_FILE, 'r') as f:
        sentences = [list(x) for x in f.readlines()]
    #model = Word2Vec(sentences, iter=100, compute_loss=True,callbacks=())
    #model.wv.save(model_path)

    wv = model
    # KeyedVectors.load(model)
    max_len = len(max(data.X, key=len))
    encoded_matrix = np.zeros(shape=(len(data.X), max_len, wv.vector_size))  # Shape: n x num_chars x vec_dim
    for i in range(len(data.X)):
        for j in range(len(data.X[i])):
            char_enc = wv[data.X[i][j]]
            encoded_matrix[i, j] = char_enc

    #print(encoded_matrix.shape)

    X = encoded_matrix
    #print(X)
    y = np.array(data.y)
    emb_size = 100

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #vectorizer, corpus_matrix, _ = get_tfidf_model(X_train)
    #X_train=X_train.numpy()
    #X_train= np.array(X_train)

    # X_train= np.ravel(X_train)
    #X_train= torch.tensor(X_train)

    X_train= np.array(X_train)
    X_train= np.reshape(X_train,(5709,6000,1))
    #X_train= torch.tensor(X_train)
    print(X_train.shape)
    print(y_train.shape)
    val_prop = 0.5  # proportion of data used for validation
    train_size = int((1 - val_prop) * X.shape[0])
#torch.tensor(x_data, dtype=torch.float32)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_data = TensorDataset(torch.tensor(X_train[:train_size], device=device,dtype=torch.double),
                               torch.tensor(y_train[:train_size], device=device,dtype=torch.double)
                                   )
    trainloader = DataLoader(train_data, batch_size=10, shuffle=True)


    input_size = 60000
    output_size = len(set(LABELS))
    num_hidden_layers = 1
    hidden_size = 100 # isn't ever used but we still set it


    seed = 24

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    mlp2 = MLP(input_size=input_size,
                                hidden_size=hidden_size,
                                num_hidden_layers=num_hidden_layers,
                                output_size=output_size)
    #print(mlp2)
    mlp2.double()
    batch_size = 1000

    #x_data_static, y_truth_static = get_toy_data(batch_size)
    #print(x_data_static)


    ##traingin
    losses = []
    batch_size = 10000
    n_batches = 10
    max_epochs = 15

    loss_change = 1.0
    last_loss = 10.0
    change_threshold = 1e-5
    epoch = 0
    all_imagefiles = []

    lr = 0.01
    optimizer = optim.Adam(params=mlp2.parameters(), lr=lr)
    cross_ent_loss = nn.CrossEntropyLoss()

    def early_termination(loss_change, change_threshold, epoch, max_epochs):
        terminate_for_loss_change = loss_change < change_threshold
        terminate_for_epochs = epoch > max_epochs

        #return terminate_for_loss_change or
        return terminate_for_epochs

    while not early_termination(loss_change, change_threshold, epoch, max_epochs):
        for _ in range(n_batches):
            # step 0: fetch the data
            #x_data, y_target = get_toy_data(batch_size)
            #vectorizer, corpus_matrix, _ = get_tfidf_model(X_train) #y_train

            # step 1: zero the gradients
            for i, (X_train, y_train) in enumerate(trainloader):
                if i % 100 == 0:
                    print("Epoch progress: %s%%" % int(100 * i / len(trainloader)))

                #trainloader
                #X_train=torch.tensor(X_train,dtype=torch.float)



                mlp2.zero_grad()

                print(X_train.shape)
                #X_train= np.array(X_train)
                #X_train= np.reshape(X_train,(10,6000,2))

                # step 2: run the forward pass
                y_pred = mlp2(X_train)#.squeeze()

                # step 3: compute the loss
                loss = cross_ent_loss(y_pred, y_train.long())

                # step 4: compute the backward pass
                loss.backward()

                # step 5: have the optimizer take an optimization step
                optimizer.step()

            # auxillary: bookkeeping
            loss_value = loss.item()
            losses.append(loss_value)
            loss_change = abs(last_loss - loss_value)
            last_loss = loss_value
'''
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        visualize_results(mlp2, x_data_static, y_truth_static, ax=ax, epoch=epoch,
                        title=f"{loss_value:0.2f}; {loss_change:0.4f}")
        plt.axis('off')
        epoch += 1
        all_imagefiles.append(f'images/mlp2_epoch{epoch}_toylearning.png')
        plt.show()
'''
'''
    knn = KNN()
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('TF-IDF + kNN accuracy:', round(acc, 4))
    print('TF-IDF + kNN AUC:', round(roc, 4))
    print('TF-IDF + kNN F1:', round(f1, 4))
    '''