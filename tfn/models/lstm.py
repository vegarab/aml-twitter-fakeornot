import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from tfn.models.model import Model
from skopt.utils import Real, Integer, Categorical
from sklearn.metrics import accuracy_score

class LSTM(nn.Module):
    def __init__(self, input_size, seq_length, hidden_size, output_size, num_layers, dropout=0.5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size * seq_length, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out_flat = lstm_out.reshape(
            (lstm_out.shape[0], -1)
        )
        dropout_out = self.dropout(lstm_out_flat)
        fc_out = self.fc(dropout_out)
        act_out = torch.sigmoid(fc_out)
        if not self.training:
            act_out = (act_out > 0.5).double()
        return act_out


class LSTMModel(Model):
    def __init__(self, num_features=50, seq_length=60, hidden_dim=50, num_layers=2, lr=0.01, momentum=0.2, dropout=0.5,
                 batch_size=50, opt='SGD'):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        output_dim = 1  # Output dimension

        # Model hyperparameters
        self.hidden_dim = hidden_dim  # Size of memory
        self.num_layers = num_layers  # No. of layers
        self.lr = lr                  # Learning rate
        self.dropout = dropout        # Dropout
        self.opt = opt
        self.momentum = momentum
        self.batch_size = batch_size

        self.model = LSTM(input_size=num_features, seq_length=seq_length, hidden_size=self.hidden_dim,
                          output_size=output_dim, num_layers=self.num_layers, dropout=self.dropout)
        self.model.double()
        if self.device == "cuda:0":
            self.model.cuda(device=self.device)


        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def fit(self, X, y, epochs=5, embedding_type='tfidf', glove=None):
        super(LSTMModel, self).fit(X, y, embedding_type, glove)
        self.model.train()

        if self.opt == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        val_prop = 0.2  # proportion of data used for validation
        train_size = int((1 - val_prop) * self.corpus_matrix.shape[0])
        if self.embedding_type == 'tfidf':
            self.corpus_matrix = self.corpus_matrix.reshape((self.corpus_matrix.shape[0],
                                                             self.corpus_matrix.shape[1],
                                                             1))
        train_data = TensorDataset(torch.tensor(self.corpus_matrix[:train_size], device=self.device),
                                   torch.tensor(y[:train_size], device=self.device)
                                   )
        val_data = TensorDataset(torch.tensor(self.corpus_matrix[train_size:], device=self.device),
                                 torch.tensor(y[train_size:], device=self.device)
                                 )

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        early_stopping = 0
        last_lr_drop = 0
        prev_training_loss = 999_999
        prev_val_loss = 999_999
        for epoch in range(epochs):
            print("Epoch: %s" % epoch)
            training_loss = 0
            val_loss = 0
            for i, (X_train, y_train) in enumerate(train_loader):
                if i % 100 == 0:
                    print("Epoch progress: %s%%" % int(100 * i / len(train_loader)))
                self.model.zero_grad()
                y_pred = self.model(X_train)
                loss = criterion(y_pred, y_train.double())
                training_loss += (loss.item() / len(train_data))
                loss.backward()
                optimizer.step()

            for (X_val, y_val) in val_loader:
                y_pred = self.model(X_val)
                loss = criterion(y_pred, y_val.double())
                val_loss += (loss.item() / len(val_data))

            print('Training Loss: %.4g' % training_loss)
            print('Validation Loss: %.4g' % val_loss)
            print('Loss / Prev : %s' % (training_loss / prev_training_loss))
            self.history['train_loss'].append(training_loss)
            self.history['val_loss'].append(val_loss)

            # if model does not improve for 3 consecutive epochs then stop early
            improvement = 1 - (val_loss / prev_val_loss)
            if improvement < 0:
                early_stopping += 1
                if early_stopping >= 2:
                    break
            else:
                improvement = 0

            # if model overfits excessively, stop early
            if training_loss < (val_loss*0.75):
                break

            # if model does not improve and optimiser is sgd, lower the learning rate
            if self.opt == 'SGD' and epochs - 2 > last_lr_drop and improvement < 0:
                self.lr /= 2
                last_lr_drop = epoch
                print("Learning rate halved.")

            prev_training_loss = training_loss
        self.val_loader = val_loader

    def predict(self, X, use_val_loader=False):
        if use_val_loader:
            test_loader = self.val_loader
        else:
            super(LSTMModel, self).predict(X)
            test_data = TensorDataset(torch.tensor(self.X_transform, device=self.device))
            test_loader = DataLoader(test_data, batch_size=self.batch_size)
        self.model.eval()
        predictions_list = []
        actual_list = []
        for (X_test, y_test) in test_loader:
            y_pred = self.model(X_test)
            if self.device == "cpu":
                predictions_list.append(y_pred.data.numpy().reshape(-1))
                actual_list.append(y_test.data.numpy().reshape(-1))
            else:
                # print('Pred:',y_pred.data.cpu().numpy().shape)
                # print('Test:',y_test.data.cpu().numpy().shape)
                predictions_list.append(y_pred.data.cpu().numpy().reshape(-1))
                actual_list.append(y_test.data.cpu().numpy().reshape(-1))

        predictions = np.hstack(predictions_list)
        actuals = np.hstack(actual_list)
        self.predictions = predictions
        self.actuals = actuals
        return predictions

    def get_val_accuracy(self):
        self.predict(None, use_val_loader=True)
        return accuracy_score(self.predictions, self.actuals)

    def get_params(self, **kwargs):
        return {}

    @classmethod
    def get_space(cls):
        return [#Categorical(['glove', 'char', 'tfidf'], name='embedding'),
                Categorical(['glove', 'char'], name='embedding'),
                # Categorical(['tfidf'], name='embedding'),
                Categorical(['SGD', 'Adam'], name='opt'),
                Categorical([50, 100, 150], name='hidden_dim'),
                Categorical([1, 2, 3], name='num_layers'),
                Categorical([1e-4, 1e-3, 1e-2], name='lr'),
                Categorical([0.5, 0.9, 0.99], name='momentum'),
                Real(0.1, 0.5, 'uniform', name='dropout')]

    def state_dict(self):
        return self.model.state_dict()

if __name__ == "__main__":
    from tfn.preprocess import Dataset
    from tfn.helper import export_results
    from tfn.feature_extraction.embedding import GloveEmbedding, CharEmbedding
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    import numpy as np

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--epochs", "-e", dest="epochs", default=50, type=int,
                        help="Maximum number of epochs to run for.")
    parser.add_argument("--emb-size", "-s", dest="emb_size", default=50, type=int,
                        help="Size of word embedding vectors (must be in 25, 50, 100, 200).")
    parser.add_argument("--emb-type", "-t", dest="type", default="glove", type=str,
                        help="Embedding type. Can be 'word' or 'char'.")
    parser.add_argument("-x", "--export-results", dest="export", action='store_true',
                        help="Exports results to results.csv")
    args = parser.parse_args()

    if args.type == "glove":
        emb_size = args.emb_size
    else:
        emb_size = 100

    # Get data GLoVe
    # data = Dataset(args.type)
    # emb = GloveEmbedding(data.X, emb_size=emb_size, type=args.type)
    # X = emb.corpus_vectors
    # y = np.array(data.y)

    # Get data char emb
    data = Dataset(args.type)
    emb = GloveEmbedding(data.X, emb_size=100, type=args.type)
    X = emb.corpus_vectors
    print(X.shape)
    y = np.array(data.y)
    emb_size = 100

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    lstm = LSTMModel(num_features=emb_size, seq_length=X.shape[1])
    lstm.fit(X_train, y_train, epochs=args.epochs)

    y_pred = lstm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('GLoVe + LSTM accuracy:', round(acc, 4))
    print('GLoVe + LSTM AUC:', round(roc, 4))
    print('GLoVe + LSTM F1:', round(f1, 4))

    if args.export:
        export_results(acc=acc, roc=roc, f1=f1)
