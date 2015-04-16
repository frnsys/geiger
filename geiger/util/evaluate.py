from sklearn import metrics
from geiger import clustering
from geiger.featurizers import Featurizer


METRICS = ['adjusted_rand', 'adjusted_mutual_info', 'completeness', 'homogeneity']


def score(strat, labels_true, labels_pred):
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
    scores = {metric: metrics.__dict__['{0}_score'.format(metric)](labels_true, labels_pred) for metric in METRICS}
    scores['strategy'] = strat.__name__
    return scores


def evaluate(docs, labels_true):
    """
    Run a few different clustering algos and see how they compare.
    """
    scores = []
    featurizer = Featurizer()
    for c in [
        clustering.hac,
        clustering.k_means,
        clustering.dbscan]:
        labels_pred = c(docs, featurizer, return_labels=True)
        scores.append(score(c, labels_true, labels_pred))

    return scores
