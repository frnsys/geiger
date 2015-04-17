from geiger.text import Vectorizer
from nytnlp.keywords import rake

class Featurizer():
    """
    Keyword featurizer.
    """

    def __init__(self):
        self.trained = False
        self.vectr = Vectorizer(min_df=0., max_df=1.)

    def featurize(self, comments, return_ctx=False):
        key_docs = []
        pseudo_docs = []
        for c in comments:
            keys = rake.extract_keywords([c.body])[0]
            # We are using cosine distance for our metric, which cannot handle empty vectors.
            # If we are using JUST these keyword docs, it's possible we have zero vectors,
            # in which case just make a pseudo-pseudo doc :)
            pseudo_doc = [k for k, count in keys] if keys else ['____<EMPTY>____']
            pseudo_docs.append(pseudo_doc)
            key_docs.append(' '.join(pseudo_doc))

        kvecs = self.vectr.vectorize(key_docs, train=not self.trained).todense()
        self.trained = True
        if return_ctx:
            return kvecs, pseudo_docs
        else:
            return kvecs
