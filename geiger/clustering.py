"""
A bunch of different clustering strategies.
"""

import numpy as np
from galaxy.cluster.ihac import Hierarchy
from sklearn.cluster import AgglomerativeClustering, KMeans
from geiger.models.lda import Model as LDA
from geiger.featurizers import featurize


def lda(comments, return_ctx=False, n_topics=None):
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


def hac(comments, return_ctx=False):
    """
    Agglomerative clustering

    to do - allow other params
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>
    """
    n = 5
    m = AgglomerativeClustering(n_clusters=n)

    if return_ctx:
        X, ctx = featurize(comments, return_ctx=True)
    else:
        X = featurize(comments)

    y = m.fit_predict(X)

    clusters = [[] for _ in range(n)]
    for i, c in enumerate(comments):
        c = c if not return_ctx else (c, ctx[i])
        clusters[y[i]].append(c)

    return clusters


def k_means(comments, return_ctx=False):
    """
    K-Means

    to do - allow other params
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>
    """
    n = 5
    m = KMeans(n_clusters=n)

    if return_ctx:
        X, ctx = featurize(comments, return_ctx=True)
    else:
        X = featurize(comments)

    y = m.fit_predict(X)

    clusters = [[] for _ in range(n)]
    for i, c in enumerate(comments):
        c = c if not return_ctx else (c, ctx[i])
        clusters[y[i]].append(c)

    return clusters


def ihac(comments, return_ctx=False, dist_cutoff=None):
    """
    Incremental hierarchical agglomerative clustering
    """
    h = Hierarchy(metric='cosine', lower_limit_scale=0.9, upper_limit_scale=1.2)

    if return_ctx:
        X, ctx = featurize(comments, return_ctx=True)
    else:
        X = featurize(comments)

    ids = h.fit(X)
    avg_distances, lvl_avgs = h.avg_distances()

    # Default cutoff, needs to be tweaked
    if dist_cutoff is None:
        dist_cutoff = np.mean(lvl_avgs)

    # Build a map of hierarchy ids to comments.
    if return_ctx:
        map = {ids[i]: (c, ctx[i]) for i, c in enumerate(comments)}
    else:
        map = {ids[i]: (c, ctx[i]) for i, c in enumerate(comments)}

    # Generate the clusters.
    clusters = h.clusters(distance_threshold=dist_cutoff, with_labels=False)

    # Get the clusters as comments.
    return [[map[id] for id in clus] for clus in clusters]
