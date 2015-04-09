import config
import importlib
from scipy import sparse
from sklearn import preprocessing


def featurize(comments, return_ctx=False):
    """
    Featurizes a list of comments according to the config.
    """
    scalr = preprocessing.StandardScaler()

    feats = []
    for name, kwargs in config.featurizers.items():
        path = '{0}.{1}'.format(__name__, name)
        mod = importlib.import_module(path)
        feats.append(mod.Featurizer(**kwargs).featurize(comments, return_ctx=return_ctx))

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

    feats = sparse.hstack(feats)
    feats = scalr.fit_transform(feats.todense())

    if return_ctx:
        return feats, ctx
    else:
        return feats
