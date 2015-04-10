from geiger.text import Vectorizer
from geiger.keywords import Rake

class Featurizer():
    """
    Keyword featurizer.
    """

    def __init__(self):
        pass

    def featurize(self, comments, return_ctx=False):
        """
        For now, using RAKE.
        <https://github.com/aneesha/RAKE>
        """
        kv = Vectorizer(min_df=0., max_df=1.)
        r = Rake('data/SmartStoplist.txt')

        key_docs = []
        pseudo_docs = []
        for c in comments:
            keys = r.run(c.body)
            # We are using cosine distance for our metric, which cannot handle empty vectors.
            # If we are using JUST these keyword docs, it's possible we have zero vectors,
            # in which case just make a pseudo-pseudo doc :)
            pseudo_doc = [k for k, count in keys] if keys else ['____<EMPTY>____']
            pseudo_docs.append(pseudo_doc)
            key_docs.append(' '.join(pseudo_doc))

        kvecs = kv.vectorize(key_docs, train=True)
        if return_ctx:
            return kvecs, pseudo_docs
        else:
            return kvecs
