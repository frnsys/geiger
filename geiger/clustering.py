"""
A bunch of different clustering strategies.
Mostly just wrappers around sklearn implementations :)
"""

from geiger.models.lda import Model as LDA
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN


def lda(comments, n_topics=None, return_ctx=False):
    """
    Cluster based on topic assignments from LDA.
    """
    m = LDA(n_topics=n_topics)
    clusters = m.cluster(comments)

    if return_ctx:
        # Create empty feature contexts
        _clusters = []
        for clus in clusters:
            _clusters.append([(mem, {}) for mem in clus])
        return _clusters

    return clusters, m


def hac(comments, featurizer, return_ctx=False, return_labels=False):
    """
    to do - allow other params
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>
    """
    m = AgglomerativeClustering(n_clusters=5)
    return _cluster(comments, m, featurizer, return_ctx=return_ctx, return_labels=return_labels)


def k_means(comments, featurizer, return_ctx=False, return_labels=False):
    """
    to do - allow other params
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>
    """
    m = KMeans(n_clusters=5)
    return _cluster(comments, m, featurizer, return_ctx=return_ctx, return_labels=return_labels)


def dbscan(comments, featurizer, return_ctx=False, return_labels=False):
    """
    to do - allow other params
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>
    """
    m = DBSCAN(metric='euclidean', eps=0.6, min_samples=3)
    return _cluster(comments, m, featurizer, return_ctx=return_ctx, return_labels=return_labels)


def _cluster(comments, model, featurizer, return_ctx=False, return_labels=False):
    """
    Capitalize on sklearn's common interfaces
    """
    if return_ctx:
        X, ctx = featurizer.featurize(comments, return_ctx=True)
    else:
        X = featurizer.featurize(comments)

    y = model.fit_predict(X)

    if return_labels:
        return y

    n = max(y) + 1

    # DBSCAN can be a real hard ass and might label every comment as noise (label=-1),
    # in which case n will be 0. So return no clusters in that case.
    if n == 0:
        return []

    # Assemble clusters from labels.
    clusters = [[] for _ in range(n)]
    for i, c in enumerate(comments):
        c = c if not return_ctx else (c, ctx[i])
        clusters[y[i]].append(c)
    return clusters
