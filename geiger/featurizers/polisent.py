from geiger.models.polisent import Model


class Featurizer():
    """
    Uses the polisent model to classify positive/negative valence.
    """
    def __init__(self):
        self.m = Model()

    def featurize(self, comments, return_ctx=False):
        feats = self.m.predict_proba(comments)
        feats = feats.reshape(feats.shape[0], 1)

        if return_ctx:
            return feats, feats
        else:
            return feats

