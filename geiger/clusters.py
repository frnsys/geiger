import math
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN


def cluster(dist_mat, eps, min_samples=3, redundant_cutoff=0.8):
    if eps is None:
        eps = estimate_eps(dist_mat)

    # Mean nearest distances
    mean_nd = np.mean(np.apply_along_axis(lambda a: np.min(a[np.nonzero(a)]), 1, dist_mat))
    print('mean nearest distance: {0}'.format(mean_nd))

    agg_clusters = {}
    scores = {}
    for e in eps:
        clusters, labels = _cluster(dist_mat, e, min_samples)

        # Num of clusters has to be at least 2 for the silhouette score to be
        # calculated
        if len(set(labels)) > 1:
            print('number of non-noise clusters: {0}'.format(len(set(labels)) - 1))
            #print('silhouette score of eps {0}'.format(e))
            #print(silhouette_score(dist_mat, labels, metric='precomputed'))
            print('dunn score')
            print(dunn(dist_mat, labels))
            print('------------------------------------------------------')
        if clusters:
            agg_clusters[e] = clusters
            scores[e] = score_clusters(clusters, dist_mat.shape[0])
            print('eps {} labels: {}'.format(e, labels))
        else:
            print('eps {} got no clusters'.format(e))

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

    # Merge redundant clusters
    final_clusters = _merge(final_clusters, redundant_cutoff=redundant_cutoff)
    return final_clusters


def _merge(clusters, redundant_cutoff=0.8):
    candidates = [set(clus) for clus in clusters]
    processed = []

    while len(candidates) > 1:
        overlapping = []

        c_i = candidates.pop()

        # Compare candidate against other candidates
        for c_j in candidates:
            # Compute Jaccard scores
            s = len(c_i.intersection(c_j)) / len(c_i.union(c_j))
            if s >= redundant_cutoff:
                overlapping.append((c_j, s))

        # If no overlapping clusters, we're done with this cluster
        if not overlapping:
            processed.append(c_i)

        # Otherwise, merge the most similar clusters as a new candidate
        else:
            c_j = max(overlapping, key=lambda x: x[1])[0]
            candidates.remove(c_j)
            candidates.append(c_i.union(c_j))

    # Add the left over candidate, if any
    processed += candidates

    # Return as lists again
    return [list(clus) for clus in processed]


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



def dunn(dist_mat, labels):
    """
    See: <https://en.wikipedia.org/wiki/Cluster_analysis#External_evaluation>
    """
    # Map indices to labels
    cluster_indices = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_indices[label].append(i)

    # Remove noise
    del cluster_indices[-1]

    if len(cluster_indices) <= 1:
        return 0.

    # Intra-cluster distance is the max distance b/w members
    intra_cds = []
    for indices in cluster_indices.values():
        sub_mat = _clus_mat(dist_mat, indices)
        intra_cds.append( np.max(sub_mat) )

    # Find the largest intra-cluster distance
    max_intra_cd = max(intra_cds)

    # Inter-cluster distance is the min distance b/w members
    inter_cds = []
    for indices1 in cluster_indices.values():
        for indices2 in cluster_indices.values():
            if indices1 == indices2:
                continue
            sub_mat = _inter_clus_mat(dist_mat, indices1, indices2)
            inter_cds.append( np.min(sub_mat) )

    # Find the smallest inter-cluster distance
    min_inter_cd = min(inter_cds)

    return min_inter_cd/max_intra_cd


def _clus_mat(dist_mat, indices):
    """
    Returns a submatrix representing the internal distance matrix for a cluster.
    """
    rows, cols = zip(*[([i], i) for i in indices])
    return dist_mat[rows, cols]


def _inter_clus_mat(dist_mat, indices1, indices2):
    """
    Returns a submatrix presenting the distance matrix between two clusters' members.
    """
    rows = [[i] for i in indices1]
    cols = indices2
    return dist_mat[rows, cols]


def estimate_eps(dist_mat, n_closest=5):
    """
    Estimates possible eps values (to be used with DBSCAN)
    for a given distance matrix by looking at the largest distance "jumps"
    amongst the `n_closest` distances for each item.

    Tip: the value for `n_closest` is important - set it too large and you may only get
    really large distances which are uninformative. Set it too small and you may get
    premature cutoffs (i.e. select jumps which are really not that big).

    TO DO this could be fancier by calculating support for particular eps values,
    e.g. 80% are around 4.2 or w/e
    """
    dist_mat = dist_mat.copy()

    # To ignore i == j distances
    dist_mat[np.where(dist_mat == 0)] = np.inf
    estimates = []
    for i in range(dist_mat.shape[0]):
        # Indices of the n closest distances
        row = dist_mat[i]
        dists = sorted(np.partition(row, n_closest)[:n_closest])
        difs = [(x,
                 y,
                 (y - x)) for x, y in zip(dists, dists[1:])]
        eps_candidate, _, jump = max(difs, key=lambda x: x[2])

        # TO DO add proper logging
        #print('~~~~~~~~~~')
        #difs_strs = [('{:.2f}'.format(x),
                      #'{:.2f}'.format(y),
                      #'{:.2f}'.format(y - x)) for x, y in zip(dists, dists[1:])]
        #print(dists)
        #print(difs_strs)
        #print(eps_candidate)
        #print(jump)
        #print('~~~~~~~~~~')

        estimates.append(eps_candidate)
        return sorted(estimates)
