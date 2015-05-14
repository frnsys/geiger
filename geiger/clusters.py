import math
import numpy as np
from sklearn.cluster import DBSCAN


def cluster(dist_mat, eps, min_samples=3):
    # Mean nearest distances
    mean_nd = np.mean(np.apply_along_axis(lambda a: np.min(a[np.nonzero(a)]), 1, dist_mat))
    print('mean nearest distance: {0}'.format(mean_nd))

    agg_clusters = {}
    scores = {}
    for e in eps:
        clusters, labels = _cluster(dist_mat, e, min_samples)
        if clusters:
            agg_clusters[e] = clusters
            scores[e] = score_clusters(clusters, dist_mat.shape[0])

    best_eps = max(scores, key=lambda k: scores[k])

    # Focus in on most promising eps
    for e in np.arange(best_eps - 0.1, best_eps + 0.1, 0.025):
        clusters, labels = _cluster(dist_mat, e, min_samples)
        if clusters:
            agg_clusters[e] = clusters
            scores[e] = score_clusters(clusters, dist_mat.shape[0])

    # Merge clustering results
    final_clusters = []
    for e, clusters in agg_clusters.items():
        if scores[e] > 0.0:
            final_clusters += [c for c in clusters if c not in final_clusters]

    return final_clusters


def _cluster(dist_mat, eps, min_samples):
    m = DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples)
    y = m.fit_predict(dist_mat)

    n = max(y) + 1

    if n == 0:
        return [], y

    else:
        clusters = [[] for _ in range(n)]
        for i in range(len(y)):
            if y[i] >= 0:
                clusters[y[i]].append(i)
        return clusters, y



def score_clusters(clusters, n):
    """
    Want to favor more evenly distributed clusters
    which cover a greater amount of the total documents.

    E.g.
        - not [300]
        - not [298, 2]
        - not [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
        - more like [20, 14, 18, 21, 8]
    """
    n_clusters = len(clusters)
    sizes = [len(c) for c in clusters]

    # How many comments are represented by the clusters
    coverage = sum(sizes)/n

    # How much coverage is accounted for by a single cluster
    gravity = math.log(sum(sizes)/max(sizes))

    # Avg discounting the largest cluster
    avg_size = (sum(sizes)-max(sizes))/len(sizes)

    return coverage * math.sqrt(gravity * avg_size) * n_clusters
