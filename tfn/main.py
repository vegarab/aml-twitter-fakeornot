from tfn.preprocess import Dataset
from tfn.helper import export_results
from tfn.data_augmentation.augmentation import AugmentWithEmbeddings
from tfn.models import CosineSimilarity, Dummy, KNN, LSTMModel, Naive_Bayes, RandomForest, SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

data_twitter = Dataset('twitter')
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(data.X, data.y)
aug = AugmentWithEmbeddings(X_train_t, y_train_t)
X_aug, y_aug = aug.X_aug, aug.y_aug

models = {
    'Dummy': Dummy(),
    'Cosine Similarity': CosineSimilarity(),
    'kNN': KNN(),
    'Naive Bayes': Naive_Bayes(),
    'Random Forest': RandomForest(),
    'SVM': SVM()
}

for name, model in models:
    for aug in [True, False]:
        if aug:
            model.fit(X_aug, y_aug)
            name = name + " (aug)"
        else:
            model.fit(X_train_t, y_train_t)
        y_pred = model.predict(X_test_t)

        acc = accuracy_score(y_test_t, y_pred)
        roc = roc_auc_score(y_test_t, y_pred)
        f1 = f1_score(y_test_t, y_pred)

        print('%s accuracy: %s' % (name, round(acc, 4)))
        print('%s AUC: %s' % (name, round(roc, 4)))
        print('%s F1: %s' % (name, round(f1, 4)))

        export_results(name=name, acc=acc, roc=roc, f1=f1)

# data_glove = Dataset('glove')
# X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(data.X, data.y)