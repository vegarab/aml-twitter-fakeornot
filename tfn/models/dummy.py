from tfn.models.model import Model
from tfn.preprocess import split_binary_classes
from tfn.feature_extraction.tf_idf import get_tfidf_model

from collections import Counter

# Baseline naive predictor
# Predicts the class with the most observations for all observations

class Dummy(Model):
    def fit(self, X, y):
        self.dummy_pred = sorted(Counter(data.y).items(), key=lambda x: x[1], reverse=True)[0][0]

    def predict(self, X):
        y_pred = [self.dummy_pred]*len(X)
        return y_pred

if __name__ == '__main__':
    from tfn.preprocess import Dataset
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    data = Dataset('twitter')

    dummy = Dummy()
    dummy.fit(data.X_train, data.y_train)

    y_pred = dummy.predict(data.X_test)

    print('Dummy accuracy:', round(accuracy_score(data.y_test, y_pred), 4))
    print('Dummy AUC:', round(roc_auc_score(data.y_test, y_pred), 4))