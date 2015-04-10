"""
A bunch of different clustering strategies.
"""

import numpy as np
from galaxy.cluster.ihac import Hierarchy
from sklearn.cluster import AgglomerativeClustering, KMeans
from geiger.models.lda import Model as LDA
from geiger.featurizers import featurize


def lda(comments, n_topics=None):
    """
    Cluster based on topic assignments from LDA.
    """
    m = LDA(n_topics=n_topics)
    #return m.cluster(comments)
    return m.cluster(comments), m


def hac(comments):
    """
    Agglomerative clustering

    to do - allow other params
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>
    """
    n = 5
    m = AgglomerativeClustering(n_clusters=n)

    X = featurize(comments)
    y = m.fit_predict(X)

    clusters = [[] for _ in range(n)]
    for i, c in enumerate(comments):
        clusters[y[i]].append(c)

    return clusters


def k_means(comments):
    """
    K-Means

    to do - allow other params
    <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>
    """
    n = 5
    m = KMeans(n_clusters=n)

    X = featurize(comments)
    y = m.fit_predict(X)

    clusters = [[] for _ in range(n)]
    for i, c in enumerate(comments):
        clusters[y[i]].append(c)

    return clusters


def ihac(comments, dist_cutoff=None):
    """
    Incremental hierarchical agglomerative clustering
    """
    h = Hierarchy(metric='cosine', lower_limit_scale=0.9, upper_limit_scale=1.2)
    X = featurize(comments)
    ids = h.fit(X)

    avg_distances, lvl_avgs = h.avg_distances()

    # Default cutoff, needs to be tweaked
    if dist_cutoff is None:
        dist_cutoff = np.mean(lvl_avgs)

    # Build a map of hierarchy ids to comments.
    map = {ids[i]: c for i, c in enumerate(comments)}

    # Generate the clusters.
    clusters = h.clusters(distance_threshold=dist_cutoff, with_labels=False)

    # Get the clusters as comments.
    return [[map[id] for id in clus] for clus in clusters]
