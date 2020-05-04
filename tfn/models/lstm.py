import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import time

from tfn.models.model import Model


class LSTM(nn.Module):
    def __init__(self, input_size, seq_length, hidden_size, output_size, num_layers, dropout=0.5):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size*seq_length, output_size)
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
    def __init__(self, num_features=50, seq_length=60):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        hidden_dim = 50 #Size of memory
        num_layers = 2  # No. of layers
        output_dim = 1  # Output dimension

        self.model = LSTM(input_size=num_features, seq_length=seq_length, hidden_size=hidden_dim, output_size=output_dim,
                          num_layers=num_layers)
        self.model.double()
        if self.device == "cuda:0":
            self.model.cuda(device=self.device)

        self.batch_size = 20

    def fit(self, X, y, epochs=5):
        self.model.train()
        learning_rate = 0.01
        momentum = 0.2

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.BCELoss()

        val_prop = 0.1 # proportion of data used for validation
        train_size = int((1-val_prop) * X.shape[0])
        train_data = TensorDataset(torch.tensor(X[:train_size], device=self.device),
                                   torch.tensor(y[:train_size], device=self.device)
            )
        val_data = TensorDataset(torch.tensor(X[train_size:], device=self.device),
                                   torch.tensor(y[train_size:], device=self.device)
            )

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        last_lr_drop = 0
        prev_training_loss = 999_999
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

            # If last learning rate drop happened more than 5 epochs to go and the current training loss
            # is higher than 99.9% of the previous, then half the learning rate.
            if epoch - 5 > last_lr_drop and (training_loss / prev_training_loss) > 0.999:
                learning_rate /= 2
                last_lr_drop = epoch
                print("Learning rate halved.")

            # End early if lr falls too low
            if learning_rate < 0.0001:
                break
            prev_training_loss = training_loss

        # Save model
        save_path = '../misc/model_save_%s' % time.time()
        torch.save(self.model.state_dict(), save_path)

    def predict(self, X):
        self.model.eval()
        test_data = TensorDataset(torch.tensor(X, device=self.device))
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        predictions_list = []
        for X_test in test_loader:
            X_test = X_test[0]
            y_pred = self.model(X_test)
            if self.device == "cpu":
                predictions_list.append(y_pred.data.numpy())
            else:
                predictions_list.append(y_pred.data.cpu().numpy())
        predictions = np.vstack(predictions_list)
        return predictions


if __name__ == "__main__":
    from tfn.preprocess import Dataset
    from tfn.helper import export_results
    from tfn.feature_extraction.embedding import GloveEmbedding
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
        emb_size = 300

    # Get data
    data = Dataset(args.type)
    emb = GloveEmbedding(data.X, emb_size=emb_size, type=args.type)
    X = emb.corpus_vectors
    y = np.array(data.y)

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