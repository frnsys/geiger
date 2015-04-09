import os
import config
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split


class Model():
    """
    A model which calculates the probability of a comment being
    of positive sentiment.
    """
    path = os.path.join(config.models_path, 'sentiment.pkl')

    def __init__(self):
        if os.path.exists(path):
            self.m = joblib.load(path)
        else:
            self.m = None

    def predict_proba(self, features):
        return self.m.predict_proba(features)[0][1]

    def train(self, X, y):
        """
        Trains the model on labeled data
        and saves it to disk.
        """
        m = linear_model.LogisticRegression('l2')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        params = {'C':[0.01, 0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}

        gs = GridSearchCV(m, params, n_jobs=-1, scoring='roc_auc')
        gs.fit(X_train, y_train)

        self.m = gs.best_estimator_.fit(X, y)
        joblib.dump(self.m, self.path)
