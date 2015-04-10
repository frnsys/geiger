from geiger.text import Vectorizer


class Featurizer():
    """
    Simple bag-of-words featurizer.
    """
    def __init__(self):
        self.trained = False

    def featurize(self, comments, return_ctx=False):
        v = Vectorizer()
        vecs = v.vectorize([c.body for c in comments], train=not self.trained)

        self.trained = True

        if return_ctx:
            return vecs, vecs
        else:
            return vecs

