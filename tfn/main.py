import warnings
warnings.filterwarnings(action='ignore',category=UserWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

from tfn import OPT_RESULTS, CHAR_TRAINING_FILE, CHAR_MODEL
from tfn.feature_extraction import embedding
from tfn.logger import log_sk_model, log_torch_model
from tfn.preprocess import Dataset
from tfn.models import CosineSimilarity, Dummy, KNN, LSTMModel, Naive_Bayes, RandomForest, SVM, GradientBoost, CNNModel

import numpy as np
from numpy.random import permutation
from argparse import ArgumentParser
import time
from tqdm import tqdm
import os
import pickle
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.callbacks import DeltaYStopper
from sklearn.model_selection import cross_val_score





def _train_test_val_aug_split(data, aug_data, num_copies, test_prop, val_prop):
    data_len = len(data.X)
    idx_shuffle = permutation(data_len)
    # Create training set (from augmented data)
    train_idx = idx_shuffle[:int(data_len * (1 - test_prop - val_prop))]
    X_aug = []
    y_aug = []
    multiplier = num_copies + 1
    for i in train_idx:
        X_aug += aug_data.X_aug[multiplier * i:multiplier * (i + 1)]
        y_aug += aug_data.y_aug[multiplier * i:multiplier * (i + 1)]
    idx_shuffle_aug = list(permutation(len(X_aug)))
    X_aug = [X_aug[i] for i in idx_shuffle_aug]
    y_aug = [y_aug[i] for i in idx_shuffle_aug]

    # Create training set (from non-augmented data)
    X_train = [data.X[i] for i in train_idx]
    y_train = [data.y[i] for i in train_idx]

    # Create validation set (from non-augmented data)
    val_idx = idx_shuffle[int(data_len * (1 - test_prop - val_prop)):int(data_len * (1 - test_prop))]
    X_val = [data.X[i] for i in val_idx]
    y_val = [data.y[i] for i in val_idx]

    # Create testing set (from non-augmented data)
    test_idx = idx_shuffle[int(data_len * (1 - test_prop)):]
    X_test = [data.X[i] for i in test_idx]
    y_test = [data.y[i] for i in test_idx]

    return X_aug, y_aug, X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-aug", "-s", dest="save_aug", default=None,
                        help="Name of augmentation save path. It is recommended that augmentation is saved whenever "
                             "run as this can take a very long time.")
    parser.add_argument("--load-aug", "-l", dest="load_aug", default=None,
                        help="Name of augmentation load path. It is recommended that augmentation is loaded from "
                             "pre-made file as re-running can take a very long time.")
    parser.add_argument("--aug-copies", "-c", dest="aug_copies", default=5, type=int,
                        help="Number of copies of the data made by augmentation (does nothing if augmentation loaded "
                             "from file).")
    parser.add_argument("--repl-prob", "-r", dest="repl_prob", default=0.25, type=float,
                        help="Chance that a word is replaced during augmentation.")
    parser.add_argument("--test-prop", "-t", dest="test_prop", default=0.2, type=float,
                        help="Proportion of data used for testing.")
    parser.add_argument("--val-prop", "-v", dest="val_prop", default=0.2, type=float,
                        help="Proportion of data used for validation.")
    parser.add_argument("--no-export", "-x", dest="no_export", action="store_false",
                        help="Results of running will not be stored in results.csv.")
    parser.add_argument("--n-calls", "-n", dest="n_calls", default=10, type=int,
                        help="Number of calls to each model for Bayesian optimisation.")
    args = parser.parse_args()

    models = {
        'CNN': CNNModel,
        # 'RandomForest': RandomForest,
        # 'LSTM': LSTMModel,
        # 'SVM': SVM,
        # 'kNN': KNN,
        # 'NaiveBayes': Naive_Bayes,
        # 'GradientBoost': GradientBoost,
        # 'CosineSimilarity': CosineSimilarity
    }
    print('Processing datasets...')
    datasets = {
        'tfidf': Dataset('twitter'),
        'glove': Dataset('glove'),
        'char': Dataset('char')
    }

    if not os.path.exists(CHAR_MODEL):
        embedding.CharEmbedding(None, train=True, training_path=CHAR_TRAINING_FILE, train_only=True)

    # Initialise glove embeddings
    glove_init = embedding.GloveEmbedding(emb_size=200)
    # Run SK models with hyperparameter search...
    print('Running non-neural approaches w/ hyperparameter tuning...')
    for model_type in models:
        t1 = time.time()
        print(f"Processing model {model_type}")
        pbar = tqdm(total=args.n_calls)
        model = models[model_type]
        space = models[model_type].get_space()

        if model_type == 'LSTM':
            @use_named_args(space)
            def objective(**params):
                print(params)
                embedding_type = params['embedding']
                if embedding_type == 'tfidf':
                    embedding_size = 1
                elif embedding_type == 'glove':
                    embedding_size = glove_init.emb_size
                else:
                    embedding_size = 100
                model_params = params.copy()
                model_params.pop('embedding', None)
                X = datasets[embedding_type].X
                y = datasets[embedding_type].y
                max_len = len(max(X, key=len))
                model = models[model_type](num_features=embedding_size, seq_length=max_len, **model_params)
                model.fit(X, y, epochs=50, embedding_type=embedding_type, glove=glove_init)
                acc = model.get_val_accuracy()
                log_torch_model(model, acc, params)
                pbar.update(1)
                return -acc
        else:
            @use_named_args(space)
            def objective(**params):
                print(params)
                embedding_type = params['embedding']
                model_params = params.copy()
                model_params.pop('embedding', None)
                X = datasets[embedding_type].X
                y = datasets[embedding_type].y
                model = models[model_type](**model_params)
                cv = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring="accuracy",
                                     fit_params={'embedding_type': embedding_type, 'glove': glove_init})
                cv_mean = np.mean(cv)
                cv_std = np.std(cv)
                log_sk_model(model, cv_mean, cv_std, params)
                pbar.update(1)
                return -cv_mean

        results = gp_minimize(objective, space, n_calls=args.n_calls, n_jobs=-1,
                              callback=DeltaYStopper(0.001, n_best=3))
        t2 = time.time()
        pbar.close()
        if not os.path.exists(OPT_RESULTS):
            os.mkdir(OPT_RESULTS)
        with open(os.path.join(OPT_RESULTS, f"{model_type}-{int(time.time())}"), 'wb') as f:
            pickle.dump(results, f)
        print(f"{model_type} done. \nBest avg accuracy score (5-fold): {-results.fun}\nTime taken: {t2-t1}")

