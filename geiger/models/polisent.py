import os
import config
import numpy as np
from sklearn.externals import joblib
from sklearn import svm, metrics, preprocessing
from sklearn.cross_validation import train_test_split
from geiger.featurizers import opinion, bow
from geiger.comment import Doc


class Model():
    """
    This is a "sentiment" model based off of Congress voting data, so
    it has some sense of a positive or negative argument.

    See <http://www.cs.cornell.edu/home/llee/data/convote.html>.

    Evaluation notes which may explain some decisions here:

        Data: convote 1.1 (used 8075 examples, 0.8/0.2 training/test split)

        bow, opinion, subjectivity (no scaling)
            LogisticRegression
                recall:     0.88066825775656321
                roc_auc:    0.85153104007519287
                accuracy:   0.85263157894736841
                precision:  0.84246575342465757
            SVC
                recall:     0.75536992840095463
                roc_auc:    0.64280722932274248
                accuracy:   0.6470588235294118
                precision:  0.63426853707414832
            RandomForestClassifier
                recall:     0.89737470167064437
                roc_auc:    0.90943381158178294
                accuracy:   0.90897832817337465
                precision:  0.92496924969249694

        bow, opinion, subjectivity (scaled)
            LogisticRegression
                recall:     0.93107476635514019
                roc_auc:    0.92403540689298513
                accuracy:   0.92445820433436532
                precision:  0.92674418604651165
            SVC
                recall:     0.95210280373831779
                roc_auc:    0.94311332545282156
                accuracy:   0.94365325077399376
                precision:  0.94219653179190754
            RandomForestClassifier
                recall:     0.91705607476635509
                roc_auc:    0.92558996096684021
                accuracy:   0.92507739938080491
                precision:  0.94011976047904189

        bow (scaled)
            LogisticRegression
                recall:     0.93246445497630337
                roc_auc:    0.93769785654132931
                accuracy:   0.93746130030959751
                precision:  0.94705174488567989
            SVC
                recall:     0.95616113744075826
                roc_auc:    0.94500663875928959
                accuracy:   0.94551083591331264
                precision:  0.94055944055944052
            RandomForestClassifier
                recall:     0.90047393364928907
                roc_auc:    0.91002944412684939
                accuracy:   0.90959752321981424
                precision:  0.92457420924574207

        bow + opinion (scaled)
            LogisticRegression
                recall:     0.92840375586854462
                roc_auc:    0.92947055421212277
                accuracy:   0.92941176470588238
                precision:  0.9372037914691943
            SVC
                recall:     0.96126760563380287
                roc_auc:    0.94852371107378219
                accuracy:   0.94922600619195041
                precision:  0.94354838709677424
            RandomForestClassifier
                recall:     0.9178403755868545
                roc_auc:    0.92418886407127787
                accuracy:   0.92383900928792573
                precision:  0.93652694610778442

        bow + subjectivity (scaled)
            LogisticRegression
                recall:     0.90794016110471809
                roc_auc:    0.91911753363546889
                accuracy:   0.91826625386996907
                precision:  0.93816884661117717
            SVC
                recall:     0.93440736478711162
                roc_auc:    0.94106427220588817
                accuracy:   0.94055727554179569
                precision:  0.95417156286721505
            RandomForestClassifier
                recall:     0.89067894131185266
                roc_auc:    0.92322150818943838
                accuracy:   0.92074303405572755
                precision:  0.95910780669144979
    """
    base_path = os.path.join(config.models_path, 'polisent')
    model_path = os.path.join(base_path, 'model.pkl')
    feats_path = os.path.join(base_path, 'featurizers.pkl')
    scalr_path = os.path.join(base_path, 'scaler.pkl')

    def __init__(self):
        if os.path.exists(self.model_path):
            self.m = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scalr_path)
            self.featurizers = joblib.load(self.feats_path)
        else:
            self.m = None
            self.scaler = None
            self.featurizers = None

    def predict_proba(self, docs):
        """
        Returns probability of the positive class for a set of documents.
        """
        X = self._featurize(docs)
        return self.m.predict_proba(X)[:,1]

    def train(self):
        """
        Trains the model on labeled data and
        saves the model, its featurizers, and its scaler.
        """
        data = [d for d in self._load_data()]
        docs, y = zip(*data)

        self.scaler = preprocessing.StandardScaler()
        self.featurizers = [opinion.Featurizer(), bow.Featurizer()]
        X = self._featurize(docs)

        # Have to toggle probability estimates so we can use `predict_proba` later on.
        # This does slow down training time quite a bit, but in theory you're only training once.
        self.m = svm.SVC(probability=True)
        self.m.fit(X, y)

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        joblib.dump(self.m, self.model_path)
        joblib.dump(self.scaler, self.scalr_path)
        joblib.dump(self.featurizers, self.feats_path)

    def evaluate(self, X_test, y_test, threshold=None):
        if threshold is None: y_pred = self.m.predict(X_test)
        else:
            y_prob = self.m.predict_proba(X_test)

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
            'model': self.m.__class__.__name__,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

    def _featurize(self, docs):
        """
        Internally generates features for use in the
        sentiment model.
        """
        feats = [f.featurize(docs) for f in self.featurizers]
        X = np.hstack(feats)
        X = self.scaler.fit_transform(X)
        return X

    def _load_data(self):
        """
        Load the convote 1.1 data for training.
        """
        data_path = 'data/sentiment/'
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
            yield from self.__load_data(path)

    def __load_data(self, path):
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
                yield (Doc(text), valence)
