import numpy as np
from sklearn import metrics
from sklearn.grid_search import ParameterGrid
from galaxy.cluster.ihac import Hierarchy
from geiger.featurizers import featurize

METRICS = ['adjusted_rand', 'adjusted_mutual_info', 'completeness', 'homogeneity']


def score(labels_true, labels_pred):
    """
    Score clustering results.

    These labels to NOT need to be congruent,
    these scoring functions only consider the cluster composition.

    That is::

        labels_true = [0,0,0,1,1,1]
        labels_pred = [5,5,5,2,2,2]
        score(labels_pred)
        >>> 1.0

    Even though the labels aren't exactly the same,
    all that matters is that the items which belong together
    have been clustered together.
    """
    return {metric: metrics.__dict__['{0}_score'.format(metric)](labels_true, labels_pred) for metric in METRICS}


def evaluate(docs, labels_true):
    features = featurize(docs)
    h = Hierarchy(metric='cosine', lower_limit_scale=0.9, upper_limit_scale=1.2)
    ids = h.fit(features)

    # Build a map of hierarchy ids to docs.
    map = {ids[i]: c for i, c in enumerate(docs)}

    pg = ParameterGrid({
        'distance_threshold': np.arange(0.1, 0.8, 0.05)
    })

    scores = []
    for params in pg:
        params['with_labels'] = True
        clusters, labels_pred = h.clusters(**params)
        print(labels_pred)
        scores.append((params, score(labels_true, labels_pred)))

    return scores
