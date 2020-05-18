import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import time
import numpy as np
import matplotlib.pyplot as plt

from tfn.models.model import Model


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
    def __init__(self, num_features=50, seq_length=60):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        hidden_dim = 50  # Size of memory
        num_layers = 2  # No. of layers
        output_dim = 1  # Output dimension

        self.model = LSTM(input_size=num_features, seq_length=seq_length, hidden_size=hidden_dim,
                          output_size=output_dim,
                          num_layers=num_layers)
        self.model.double()
        if self.device == "cuda:0":
            self.model.cuda(device=self.device)

        self.batch_size = 20

    def fit(self, X, y, epochs=5, lr=0.01, momentum=0.2):
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.BCELoss()

        val_prop = 0.1  # proportion of data used for validation
        train_size = int((1 - val_prop) * X.shape[0])
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

        val_losses =  np.ndarray(epochs)
        val_accs = np.ndarray(epochs)
        train_losses = np.ndarray(epochs)
        train_accs = np.ndarray(epochs)
        for epoch in range(epochs):
            print("Epoch: %s" % epoch)
            training_loss = 0
            train_acc = []
            val_loss = 0
            val_acc = []
            for i, (X_train, y_train) in enumerate(train_loader):
                if i % 100 == 0:
                    print("Epoch progress: %s%%" % int(100 * i / len(train_loader)))
                self.model.zero_grad()
                y_pred = self.model(X_train)
                loss = criterion(y_pred, y_train.double())
                training_loss += (loss.item() / len(train_data))
                loss.backward()
                optimizer.step()

                # Convert to integer predictions
                y_pred = np.where(y_pred.cpu().detach().numpy() > 0.5, 1, 0)

                train_acc.append(accuracy_score(y_train.cpu().detach().numpy(), y_pred))

            train_accs[epoch] = sum(train_acc)/len(train_acc)

            for (X_val, y_val) in val_loader:
                y_pred = self.model(X_val)
                loss = criterion(y_pred, y_val.double())
                val_loss += (loss.item() / len(val_data))

                # Convert to integer predictions
                y_pred = np.where(y_pred.cpu().detach().numpy() > 0.5, 1, 0)
                val_acc.append(accuracy_score(y_val.cpu().detach().numpy(), y_pred))

            val_accs[epoch] = sum(val_acc)/len(val_acc)

            print('Training Loss: %.4g' % training_loss)
            train_losses[epoch] = training_loss
            print('Validation Loss: %.4g' % val_loss)
            val_losses[epoch] = val_loss
            print('Loss / Prev : %s' % (training_loss / prev_training_loss))
            

            # If last learning rate drop happened more than 5 epochs to go and the current training loss
            # is higher than 99.9% of the previous, then half the learning rate.
            if epoch - 5 > last_lr_drop and (training_loss / prev_training_loss) > 0.999:
                lr /= 2
                last_lr_drop = epoch
                print("Learning rate halved.")

            # End early if lr falls too low
            if lr < 0.0001:
                break
            prev_training_loss = training_loss

        # Save model
        try:
            save_path = '../misc/model_save_%s' % time.time()
            torch.save(self.model.state_dict(), save_path)
        except Exception as e:
            print("Could not save file:", e)

        return train_losses, train_accs, val_losses, val_accs

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


def visualize_training(hist1, label1, hist2, label2, xlabel, ylabel, filepath):
    plt.figure(figsize=(8,8))
    plt.plot(hist1, label=label1)
    plt.plot(hist2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(filepath)
    plt.show()

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
                        help="Embedding type. Can be 'glove' or 'char'.")
    parser.add_argument("-x", "--export-results", dest="export", action='store_true',
                        help="Exports results to results.csv")
    args = parser.parse_args()

    if args.type == "glove":
        emb_size = args.emb_size
        data = Dataset(args.type)
        emb = GloveEmbedding(data.X, emb_size=emb_size, type=args.type)
        X = emb.corpus_vectors
        y = np.array(data.y)
    elif args.type == "char":
        data = Dataset(args.type)
        emb = CharEmbedding(data.X)
        X = emb.X_enc
        y = np.array(data.y)
        emb_size = 100

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    lstm = LSTMModel(num_features=emb_size, seq_length=X.shape[1])
    train_losses, train_accs, val_losses, val_accs = lstm.fit(X_train, y_train, epochs=args.epochs)

    visualize_training(train_losses, 'Training loss', 
                       val_losses, 'Validation loss',
                       'Epochs', 'Binary Cross-entropy Loss', 'loss_lstm.png')
    visualize_training(train_accs, 'Training accuracy', 
                       val_accs, 'Validation accuracy',
                       'Epochs', 'Prediction Accuracy', 'acc_lstm.png')

    y_pred = lstm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print('GLoVe + LSTM accuracy:', round(acc, 4))
    print('GLoVe + LSTM AUC:', round(roc, 4))
    print('GLoVe + LSTM F1:', round(f1, 4))

    if args.export:
        export_results(acc=acc, roc=roc, f1=f1)
