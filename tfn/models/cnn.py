import torch
import torch.nn.functional as F

from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torchbearer

import numpy as np

from tfn.preprocess import Dataset
from tfn.models.model import Model
from tfn.feature_extraction.embedding import GloveEmbedding


N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
BATCH_SIZE=128


class CNN(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, out_channels = n_filters,
                                              kernel_size = (fs, embedding_dim))
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
    def __init__(self, n_filters, filter_sizes, output_dim, dropout, batch_size):
        super().__init__()
        
        self.model = CNN(n_filters, filter_sizes, output_dim, dropout)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        #self.device = "cpu"

        self.model.double()
        if self.device == "cuda:0":
            self.model.cuda(device=self.device)

        self.batch_size = batch_size

    def fit(self, X, y, optimiser, epochs=30):
        self.model.train()

        criterion = nn.BCELoss()

        val_prop = 0.1
        train_size = int((1 - val_prop) * X.shape[0])
        self.train_data = TensorDataset(torch.tensor(X[:train_size], device=self.device),
                                        torch.tensor(y[:train_size], device=self.device)
                                       )
        self.val_data = TensorDataset(torch.tensor(X[train_size:], device=self.device),
                                      torch.tensor(y[train_size:], device=self.device)
                                     )

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            print("Epoch %s" % epoch)
            training_loss = 0.0
            val_loss = 0.0
            for i, (X_train, y_train) in enumerate(self.train_loader):
                self.model.zero_grad()
                y_pred = self.model(X_train)
                loss = criterion(y_pred.squeeze(1), y_train.double())
                training_loss += (loss.item() / len(self.train_data))
                loss.backward()
                optimiser.step()

            for (X_val, y_val) in self.val_loader:
                y_pred = self.model(X_val)
                loss = criterion(y_pred.squeeze(1), y_val.double())
                val_loss += (loss.item() / len(self.val_data))

            print('Training Loss: %.4g' % training_loss)
            print('Validation Loss: %.4g' % val_loss)

    def predict(self, X):
        self.model.eval()
        self.pred_test_data = TensorDataset(torch.tensor(X, device=self.device))
        self.pred_test_loader = DataLoader(self.pred_test_data, batch_size=self.batch_size)
        predictions_list = []
        for X_test in self.pred_test_loader:
            X_test = X_test[0]
            y_pred = self.model(X_test)
            if self.device == "cpu":
                predictions_list.append(y_pred.data.numpy())
            else:
                predictions_list.append(y_pred.data.cpu().numpy())
        predictions = np.vstack(predictions_list)
        return predictions


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    data = Dataset("glove")
    embedding = GloveEmbedding(data.X, emb_size=50, type="glove")
    X = embedding.corpus_vectors
    y = np.array(data.y)

    cnn = CNNModel(N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, BATCH_SIZE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    #optimiser = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
    optimiser = optim.Adam(cnn.model.parameters())
    cnn.fit(X_train, y_train, optimiser, epochs=30)

    y_pred = cnn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('GLoVe + CNN accuracy:', round(acc, 4))
    print('GLoVe + CNN AUC:', round(roc, 4))
    print('GLoVe + CNN F1:', round(f1, 4))
