import config
import importlib
import numpy as np
from scipy import sparse
from sklearn import preprocessing


class Featurizer():
    """
    This is a super-featurizer,
    in that it is the amalgamation of many featurizers
    and provides a single interface.

    This way, featurizers can be cached and reused later on.

    The featurizers that are used are specified in `config.py`.

    Note that featurizers MUST return dense matrices, not a sparse ones.
    """
    def __init__(self):
        self.scalr = None
        self.featurizers = []
        for name, kwargs in config.featurizers.items():
            path = '{0}.{1}'.format(__name__, name)
            mod = importlib.import_module(path)
            self.featurizers.append(mod.Featurizer(**kwargs))

    def featurize(self, comments, return_ctx=False):
        feats = []
        for featurizer in self.featurizers:
            feats.append(featurizer.featurize(comments, return_ctx=return_ctx))

        # Extract the context for each comment.
        # This is a dictionary mapping of feature names
        # to some human-readable representation of the features
        # for each comment.
        if return_ctx:
            ctx = []
            for i, c in enumerate(comments):
                d = {}
                for j, name in enumerate(config.featurizers.keys()):
                    d[name] = feats[j][1][i]
                ctx.append(d)
            feats = [f[0] for f in feats]

        feats = np.hstack(feats)
        feats = np.nan_to_num(feats)

        if self.scalr is None:
            self.scalr = preprocessing.StandardScaler()
            feats = self.scalr.fit_transform(feats)
        else:
            feats = self.scalr.transform(feats)

        # Attach features to comments for re-use later.
        for i, c in enumerate(comments):
            c.features = feats[i]

        if return_ctx:
            return feats, ctx
        else:
            return feats
