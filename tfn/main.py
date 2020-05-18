from tfn.preprocess import Dataset
from tfn.helper import export_results
from tfn.data_augmentation.augmentation import AugmentWithEmbeddings
from tfn.models import CosineSimilarity, Dummy, KNN, LSTMModel, Naive_Bayes, RandomForest, SVM, GradientBoost
from tfn import AUG_PATH
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from numpy.random import permutation
from argparse import ArgumentParser

from skopt import gp_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import numpy as np
from tfn.feature_extraction.tf_idf import get_tfidf_model
from tfn.feature_extraction import embedding
from tfn.logger import log_sk_model
import time
from tqdm import tqdm


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
    parser.add_argument("--n-calls", "-n", dest="n_calls", default=20,
                        help="Number of calls to each model for Bayesian optimisation.")
    args = parser.parse_args()

    models = {
        'SVM': SVM,
        'RandomForest': RandomForest,
        'kNN': KNN,
        'NaiveBayes': Naive_Bayes
    }

    data = Dataset('twitter')

    # Run SK models with hyperparameter search...
    for model_type in models:
        t1 = time.time()
        print(f"Processing model {model_type}")
        pbar = tqdm(total=args.n_calls)
        space = models[model_type]().space
        model = models[model_type]

        @use_named_args(space)
        def objective(**params):
            model = models[model_type](**params)
            cv = cross_val_score(model, data.X, data.y, cv=5, n_jobs=-1, scoring="f1")
            cv_mean = np.mean(cv)
            cv_std = np.std(cv)
            log_sk_model(model, cv_mean, cv_std, params)
            pbar.update(1)
            return -cv_mean

        results = gp_minimize(objective, space, n_calls=args.n_calls)
        t2 = time.time()
        pbar.close()
        print(f"{model_type} done. \nBest avg F1 score (5-fold): {-results.fun}\nTime taken: {t2-t1}")
