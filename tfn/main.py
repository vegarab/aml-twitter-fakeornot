from tfn.preprocess import Dataset
from tfn.helper import export_results
from tfn.data_augmentation.augmentation import AugmentWithEmbeddings
from tfn.models import CosineSimilarity, Dummy, KNN, LSTMModel, Naive_Bayes, RandomForest, SVM, GradientBoost
from tfn import AUG_PATH
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from numpy.random import permutation
from argparse import ArgumentParser
import pickle


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
    parser.add_argument("--no-export", "-n", dest="no_export", action="store_false",
                        help="Results of running will not be stored in results.csv.")
    args = parser.parse_args()

    data_t = Dataset('twitter')

    # Load augmentation data
    if args.load_aug:
        with open(AUG_PATH / args.load_aug, 'rb') as aug_file:
            num_copies, aug_t = pickle.load(aug_file)
    else:
        # Run data augmentation (takes a long time)
        num_copies = args.aug_copies
        aug_t = AugmentWithEmbeddings(data_t.X, data_t.y, num_copies=num_copies, replace_pr=args.repl_prob)
        if args.save_aug:
            with open(AUG_PATH / args.save_aug, 'wb') as aug_file:
                pickle.dump((num_copies, aug_t), aug_file)

    # Assert train/test/validation proportions
    test_prop = args.test_prop
    val_prop = args.val_prop
    X_aug_t, y_aug_t, X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t = _train_test_val_aug_split(
        data_t, aug_t, num_copies, test_prop, val_prop
    )

    # Declare all models to be tested
    models = {
        'Dummy': Dummy(),
        'Cosine Similarity': CosineSimilarity(),
        'kNN': KNN(),
        'Naive Bayes': Naive_Bayes(),
        'Random Forest': RandomForest(),
        'SVM': SVM(),
        'Gradient Boosting': GradientBoost()
    }

    for model in models:
        for aug in [True, False]:
            if aug:
                models[model].fit(X_aug_t, y_aug_t)
                name = model + " (aug)"
            else:
                models[model].fit(X_train_t, y_train_t)
                name = model
            y_pred = models[model].predict(X_test_t)

            acc = accuracy_score(y_test_t, y_pred)
            roc = roc_auc_score(y_test_t, y_pred)
            f1 = f1_score(y_test_t, y_pred)

            print('%s accuracy: %s' % (name, round(acc, 4)))
            print('%s AUC: %s' % (name, round(roc, 4)))
            print('%s F1: %s' % (name, round(f1, 4)))

            if not args.no_export:
                export_results(name=name, acc=acc, roc=roc, f1=f1)

# data_glove = Dataset('glove')
# X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(data.X, data.y)
