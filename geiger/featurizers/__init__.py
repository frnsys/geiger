import config
import importlib
from scipy import sparse
from sklearn import preprocessing


# For development, cache featurizers are part of the module.
# If we start running this as part of a server, then we can't do this;
# we would need featurizers for each article instead.
featurizers = []
scalr = None
for name, kwargs in config.featurizers.items():
    path = '{0}.{1}'.format(__name__, name)
    mod = importlib.import_module(path)
    featurizers.append(mod.Featurizer(**kwargs))

def featurize(comments, return_ctx=False):
    """
    Featurizes a list of comments according to the config.
    """
    global scalr
    feats = []
    for featurizer in featurizers:
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

    feats = sparse.hstack(feats)

    if scalr is None:
        scalr = preprocessing.StandardScaler()
        feats = scalr.fit_transform(feats.todense())
    else:
        feats = scalr.transform(feats.todense())

    # Attach features to comments for re-use later.
    for i, c in enumerate(comments):
        c.features = feats[i]

    if return_ctx:
        return feats, ctx
    else:
        return feats
