from geiger.text import Vectorizer


class Featurizer():
    """
    Simple bag-of-words featurizer.
    """
    def __init__(self):
        self.trained = False
        self.vectr = Vectorizer()

    def featurize(self, comments, return_ctx=False):
        vecs = self.vectr.vectorize([c.body for c in comments], train=not self.trained).todense()
        self.trained = True

        if return_ctx:
            return vecs, vecs
        else:
            return vecs

