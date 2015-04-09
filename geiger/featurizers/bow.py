from geiger.text import Vectorizer, strip_tags

class Featurizer():
    """
    Simple bag-of-words featurizer.
    """
    def __init__(self):
        pass

    def featurize(self, comments, return_ctx = False):
        v = Vectorizer()
        vecs = v.vectorize([strip_tags(c.body) for c in comments], train=True)

        if return_ctx:
            return vecs, vecs
        else:
            return vecs

