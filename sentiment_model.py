"""
Try building sentiment models.

Data is in `data/sentiment/`.

Trying:

- logistic regression
- SVM
"""

import os
import numpy as np
import pandas as pd
from geiger.text import strip_tags
from geiger.featurizers import opinion, subjectivity
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn import metrics

data_path = 'data/sentiment/'


class Model():
    def __init__(self):
        self._m = linear_model.LogisticRegression('l2', class_weight={1: 1})

    def train(self, X_train, y_train):
        self._m.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, threshold=None):
        if threshold is None: y_pred = self._m.predict(X_test)
        else:
            y_prob = self._m.predict_proba(X_test)

            # Get predictions for the positive (1) class.
            y_pred = y_prob[:,1]
            y_pred[y_pred > threshold] = 1
            y_pred[y_pred != 1] = 0
            y_pred = y_pred.astype(int)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        roc_auc = metrics.roc_auc_score(y_test, y_pred)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'coef': list(self._m.coef_[0])
        }


def load_bbc():
    path = os.path.join(data_path, '6humanCodedDataSets/bbc1000.txt')

    # tab-separated
    with open(path, 'r') as f:
        next(f) # skip header
        for line in f:
            pos, neg, text = line.split('\t')
            text = strip_tags(text.strip())

            # [-1, 1] = neutral
            # >= 2 = positive
            # <= -2 = negative
            valence = int(pos) - int(neg)
            if valence >= 2:
                label = 1
            elif valence <= 2:
                label = 0
            else:
                continue
            yield (text, label)


def load_movies():
    """
    I don't think a sentiment model trained of movie review data
    will be useful for the domain of news comments, but
    this can function as a sanity check.
    """
    rel_paths = [
        'aclImdb/test/',
        'aclImdb/train/'
    ]
    for rel_path in rel_paths:
        path = os.path.join(data_path, rel_path)
        yield from _load_movies(path)

def _load_movies(path):
    pos_path = os.path.join(path, 'pos')
    neg_path = os.path.join(path, 'neg')

    for fname in os.listdir(pos_path):
        with open(os.path.join(pos_path, fname), 'r') as f:
            text = f.read()
            yield (text, 1)

    for fname in os.listdir(neg_path):
        with open(os.path.join(neg_path, fname), 'r') as f:
            text = f.read()
            yield (text, 0)

def load_congress():
    rel_paths = [
        'convote_v1.1/data_stage_one/training_set/',
        'convote_v1.1/data_stage_one/test_set/',
        'convote_v1.1/data_stage_two/training_set/',
        'convote_v1.1/data_stage_two/test_set/',
        'convote_v1.1/data_stage_three/training_set/',
        'convote_v1.1/data_stage_three/test_set/'
    ]

    for rel_path in rel_paths:
        path = os.path.join(data_path, rel_path)
        yield from _load_congress(path)

def _load_congress(path):
    for fname in os.listdir(path):
        with open(os.path.join(path, fname), 'r') as f:
            text = f.read()
            if len(text) <= 500:
                continue

            vote = fname.split('.')[0][-1]
            if vote == 'Y':
                valence = 1
            else:
                valence = 0
            yield (text, valence)


def main():
    opinion_f = opinion.Featurizer()
    subject_f = subjectivity.Featurizer()

    class Doc():
        def __init__(self, doc):
            self.body = doc

    data = [d for d in load_congress()]
    docs, labels = zip(*data)

    docs = [Doc(d) for d in docs]
    print('{0} examples.'.format(len(docs)))

    print('Building opinion features...')
    opinion_feats = opinion_f.featurize(docs)
    print('Building subjectivity features...')
    subject_feats = subject_f.featurize(docs)
    X = np.hstack([opinion_feats, subject_feats])

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

    m = Model()
    m.train(X_train, y_train)
    scores = m.evaluate(X_test, y_test)
    print(scores)



if __name__ == '__main__':
    main()
