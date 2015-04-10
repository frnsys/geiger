import numpy as np
from geiger.models.doc2vec import Model as Doc2Vec


class Featurizer():
    """
    Featurizes documents using a trained Doc2Vec model.
    """
    def __init__(self):
        self.m = Doc2Vec()

    def featurize(self, comments, return_ctx=False):
        print('Featurizing doc2vec...')
        docs = [c.body for c in comments]
        vecs = np.vstack([self.m.infer_vector(d) for d in docs])

        if return_ctx:
            return vecs, vecs
        else:
            return vecs
