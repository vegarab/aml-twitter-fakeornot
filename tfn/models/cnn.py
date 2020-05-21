import torch
import torch.nn.functional as F

from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np

from tfn.preprocess import Dataset
from tfn.models.model import Model
from tfn.feature_extraction.embedding import GloveEmbedding

from skopt.utils import Real, Integer, Categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(CNN, self).__init__()

        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1, out_channels = n_filters,
                                              kernel_size=(fs, embedding_dim))
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.unsqueeze(1)

        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))

        fc_out = self.fc(cat)
        act_out = torch.sigmoid(fc_out)
        if not self.training:
            act_out = (act_out > 0.5).double()

        return act_out


class CNNModel(Model):
    def __init__(self, num_features, seq_length, n_filters=100, filter_sizes=(3,4,5), output_dim=1, dropout=0.5,
                 batch_size=50, lr=0.01, momentum=0.2, opt="SGD"):
        super().__init__()
        
        self.model = CNN(num_features, n_filters, filter_sizes, output_dim, dropout)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.max_len = seq_length

        # Model hyperparameters
        self.embedding_dim = num_features
        self.n_filters = n_filters
        self.lr = lr  # Learning rate
        self.dropout = dropout  # Dropout
        self.opt = opt
        self.momentum = momentum
        self.batch_size = batch_size

        self.model.double()
        if self.device == "cuda:0":
            self.model.cuda(device=self.device)

    def fit(self, X, y, epochs=30, embedding_type='tfidf', glove=None):
        super(CNNModel, self).fit(X, y, embedding_type, glove)
        self.model.train()

        criterion = nn.BCELoss()
        if self.opt == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        val_prop = 0.2
        train_size = int((1 - val_prop) * self.corpus_matrix.shape[0])
        if self.corpus_matrix.shape[1] != self.max_len:
            zeros = np.zeros(
                shape=(self.corpus_matrix.shape[0], self.max_len, self.corpus_matrix.shape[2])
            )
            zeros[:, :self.corpus_matrix.shape[1], :] = self.corpus_matrix
            self.corpus_matrix = zeros
        self.train_data = TensorDataset(torch.tensor(self.corpus_matrix[:train_size], device=self.device),
                                        torch.tensor(y[:train_size], device=self.device)
                                       )
        self.val_data = TensorDataset(torch.tensor(self.corpus_matrix[train_size:], device=self.device),
                                      torch.tensor(y[train_size:], device=self.device)
                                     )

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)

        early_stopping = 0
        last_lr_drop = 0
        prev_training_loss = 999_999
        prev_val_loss = 999_999

        val_losses = np.ndarray(epochs)
        val_accs = np.ndarray(epochs)
        train_losses = np.ndarray(epochs)
        train_accs = np.ndarray(epochs)
        for epoch in range(epochs):
            print("Epoch %s" % epoch)
            training_loss = 0
            train_acc = []
            val_loss = 0
            val_acc = []
            for i, (X_train, y_train) in enumerate(self.train_loader):
                if i % 100 == 0:
                    print("Epoch progress: %s%%" % int(100 * i / len(self.train_loader)))
                self.model.zero_grad()
                y_pred = self.model(X_train)
                loss = criterion(y_pred.squeeze(1), y_train.double())
                training_loss += (loss.item() / len(self.train_data))
                loss.backward()
                optimizer.step()

                # Convert to integer predictions
                y_pred = np.where(y_pred.cpu().detach().numpy() > 0.5, 1, 0)

                train_acc.append(accuracy_score(y_train.cpu().detach().numpy(), y_pred))
            train_accs[epoch] = sum(train_acc) / len(train_acc)

            for (X_val, y_val) in self.val_loader:
                y_pred = self.model(X_val)
                loss = criterion(y_pred.squeeze(1), y_val.double())
                val_loss += (loss.item() / len(self.val_data))

                # Convert to integer predictions
                y_pred = np.where(y_pred.cpu().detach().numpy() > 0.5, 1, 0)
                val_acc.append(accuracy_score(y_val.cpu().detach().numpy(), y_pred))

            val_accs[epoch] = sum(val_acc) / len(val_acc)
            print('Training Loss: %.4g' % training_loss)
            train_losses[epoch] = training_loss
            print('Validation Loss: %.4g' % val_loss)
            val_losses[epoch] = val_loss
            print('Loss / Prev : %s' % (training_loss / prev_training_loss))

            # if model does not improve for 3 consecutive epochs then stop early
            improvement = 1 - (val_loss / prev_val_loss)
            if improvement < 0:
                early_stopping += 1
                if early_stopping >= 2:
                    break
            else:
                improvement = 0

            # if model overfits excessively, stop early
            if training_loss < (val_loss * 0.75):
                break

            # if model does not improve and optimiser is sgd, lower the learning rate
            if self.opt == 'SGD' and epochs - 2 > last_lr_drop and improvement < 0:
                self.lr /= 2
                last_lr_drop = epoch
                print("Learning rate halved.")

            prev_training_loss = training_loss

        return train_losses, train_accs, val_losses, val_accs

    def predict(self, X):
        super(CNNModel, self).predict(X)
        if self.X_transform.shape[1] != self.max_len:
            zeros = np.zeros(
                shape=(self.X_transform.shape[0], self.max_len, self.X_transform.shape[2])
            )
            zeros[:, :self.X_transform.shape[1], :] = self.X_transform
            self.X_transform = zeros
        test_data = TensorDataset(torch.tensor(self.X_transform, device=self.device))
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        self.model.eval()

        predictions_list = []

        for X_test in test_loader:
            X_test = X_test[0]
            y_pred = self.model(X_test)
            if self.device == "cpu":
                predictions_list.append(y_pred.data.numpy().reshape(-1))

            else:
                predictions_list.append(y_pred.data.cpu().numpy().reshape(-1))

        predictions = np.hstack(predictions_list)

        self.predictions = predictions

        return predictions

    def get_val_accuracy(self):
        self.predict(None, use_val_loader=True)
        return accuracy_score(self.predictions, self.actuals)

    def get_params(self, **kwargs):
        return {}

    @classmethod
    def get_space(cls):
        return [Categorical(['glove', 'char'], name='embedding'),
                Categorical(['SGD', 'Adam'], name='opt'),
                Categorical([1e-4, 1e-3, 1e-2], name='lr'),
                Categorical([0.5, 0.9, 0.99], name='momentum'),
                Categorical([(4,4,4), (3,4,5), (5,4,3),
                             (6,6,6), (5,6,7), (7,6,5)], name='filter_sizes'),
                Integer(50, 200, name="n_filters"),
                Real(0.1, 0.5, 'uniform', name='dropout')]
    def state_dict(self):
        return self.model.state_dict()

def visualize_training(hist1, label1, hist2, label2, xlabel, ylabel, filepath):
    plt.figure(figsize=(8, 8))
    plt.plot(hist1, label=label1)
    plt.plot(hist2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(filepath)
    plt.show()

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--epochs", "-e", dest="epochs", default=50, type=int,
                        help="Maximum number of epochs to run for.")
    parser.add_argument("--emb-size", "-s", dest="emb_size", default=50, type=int,
                        help="Size of word embedding vectors (must be in 25, 50, 100, 200).")
    parser.add_argument("--embedding", dest="embedding", default="glove")
    parser.add_argument("--opt", dest="opt", default="SGD")
    parser.add_argument("--lr", dest="lr", default=1e-3, type=float)
    parser.add_argument("--momentum", dest="momentum", default=0.5, type=float)
    parser.add_argument("--dropout", dest="dropout", default=0.5, type=float)
    parser.add_argument("--n-filters", dest="n_filters", default=100, type=int)
    parser.add_argument("--filter-sizes", dest="n_filters", nargs='+', default=[3,3,3], type=int, required=True)
    parser.add_argument("--cv", dest="cv", action="store_true")

    args = parser.parse_args()

    embedding_type = args.embedding
    data = Dataset(embedding_type)
    if embedding_type == "glove":
        emb_size = args.emb_size
        emb = GloveEmbedding(data.X, emb_size=emb_size)
    else:
        emb_size = 100
        emb = None

    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, shuffle=True)
    max_len = len(max(data.X, key=len))
    cnn = CNNModel(num_features=emb_size, seq_length=max_len)
    if not args.cv:
        train_losses, train_accs, val_losses, val_accs = cnn.fit(X_train, y_train, epochs=args.epochs,
                                                                  embedding_type=embedding_type, glove=emb)
        visualize_training(train_losses, 'Training loss',
                           val_losses, 'Validation loss',
                           'Epochs', 'Binary Cross-entropy Loss', 'loss_lstm.png')
        visualize_training(train_accs, 'Training accuracy',
                           val_accs, 'Validation accuracy',
                           'Epochs', 'Prediction Accuracy', 'acc_lstm.png')

        y_pred = cnn.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print('GLoVe + CNN accuracy:', round(acc, 4))
        print('GLoVe + CNN AUC:', round(roc, 4))
        print('GLoVe + CNN F1:', round(f1, 4))
    else:
        kf = KFold(n_splits=5)
        cv = []
        for ix, (train_index, test_index) in enumerate(kf.split(X_train)):
            print('Fold %d' % ix)
            X_t, y_t = map(lambda i: X_train[i], train_index), map(lambda i: y_train[i], train_index)
            X_t_t, y_t_t = map(lambda i: X_train[i], test_index), map(lambda i: y_train[i], test_index)
            cnn = CNNModel(num_features=emb_size, seq_length=max_len)
            cnn.fit(X_t, y_t, epochs=args.epochs,
                    embedding_type=embedding_type, glove=emb)
            y_pred = cnn.predict(X_t_t)

            acc = accuracy_score(y_t_t, y_pred)
            roc = roc_auc_score(y_t_t, y_pred)
            f1 = f1_score(y_t_t, y_pred)
            print(acc)
            print(roc)
            print(f1)
            cv.append(acc)
        cv_mean = np.mean(cv)
        cv_std = np.std(cv)
        params = {
            "embedding": args.embedding,
            "opt": args.opt,
            "lr": args.lr,
            "momentum": args.momentum,
            "dropout": args.dropout,
            "n_filters": args.n_filters,
            "filter_sizes": args.filter_sizes
        }
        log_torch_model(cnn, cv_mean, params, std=cv_std)

