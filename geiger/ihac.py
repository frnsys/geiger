"""
Geiger clusters comments to generate a _gist_ of what's happening.
"""

import numpy as np
from scipy import sparse
from galaxy.cluster.ihac import Hierarchy
from statistics import mode, StatisticsError
from geiger.featurizers import featurize


def highlights(comments, min_size=5, dist_cutoff=None):
    """
    This takes a list of Comments,
    clusters them, and then returns representatives from clusters above
    some threshold size.

    Args:
        | comments      -- list of Comments
        | min_size      -- int, minimium cluster size to consider
        | dist_cutoff   -- float, the distances at which to snip the hierarchy for clusters

    Future improvements:
        - Persist hierarchies instead of rebuilding from scratch (using Hierarchy.load & Hierarchy.save)
        - Tweak min_size and dist_cutoff for the domain.
    """

    features = featurize(comments)

    print('Clustering {0} comments...'.format(features.shape[0]))

    # Build the hierarchy.
    h = Hierarchy(metric='cosine', lower_limit_scale=0.9, upper_limit_scale=1.2)
    ids = h.fit(features)

    avg_distances, lvl_avgs = h.avg_distances()

    # Default cutoff, needs to be tweaked
    if dist_cutoff is None:
        dist_cutoff = np.mean(lvl_avgs)

    print('Processing resulting clusters (cutoff={0})...'.format(dist_cutoff))

    # Build a map of hierarchy ids to comments.
    map = {ids[i]: c for i, c in enumerate(comments)}

    # Generate the clusters.
    clusters = h.clusters(distance_threshold=dist_cutoff, with_labels=False)

    # Filter to clusters of at least some minimum size.
    clusters = [c for c in clusters if len(c) >= min_size]

    print('Filtered to {0} clusters.'.format(len(clusters)))

    # Get the clusters as comments.
    clusters = [[map[id] for id in clus] for clus in clusters]

    # From each cluster, pick the comment with the highest score.
    highlights = [(max(clus, key=lambda c: c.score), len(clus)) for clus in clusters]

    # Suppress replies, show only top-level.
    for h in highlights:
        h[0].replies = []

    # For dev purposes
    sizes = [len(c) for c in clusters] if clusters else [0] # to prevent division by zero
    try:
        mode_size = mode(sizes)
    except StatisticsError:
        mode_size = None
    stats = {
        'n_comments': len(comments),
        'n_clusters': len(sizes),
        'n_filtered_clusters': len(clusters),
        'min_size': min_size,
        'avg_cluster_size': sum(sizes)/len(sizes),
        'max_cluster_size': max(sizes),
        'min_cluster_size': min(sizes),
        'mode_cluster_size': mode_size,
        'cutoff': dist_cutoff,
        'avg_cluster_distances': avg_distances,
        'max_cluster_distances': max(lvl_avgs),
        'min_cluster_distances': min(lvl_avgs),
    }
    print(stats)

    return highlights, stats


def examine(comments):
    """
    Clusters the comments and visualizes them in a nice explorable HTML format.
    """
    features, ctx = featurize(comments, return_ctx=True)

    h = Hierarchy(metric='cosine', lower_limit_scale=0.9, upper_limit_scale=1.2)
    ids = h.fit(features)
    comments = {ids[i]: {'comment': c, 'context': ctx[i]} for i, c in enumerate(comments)}

    # Build a nested structure to represent the hierarchy.
    tree = [n for n in _build_tree(h, [h.g.root])]

    avg_distances, lvl_avgs = h.avg_distances()
    stats = {
        'n_comments': len(comments),
        'avg_cluster_distances': avg_distances,
        'max_cluster_distances': max(lvl_avgs),
        'min_cluster_distances': min(lvl_avgs),
    }

    return comments, tree, stats


def _build_tree(h, level):
    """
    Recursively build a tree starting from the specified node ids.

    Returned node representations are tuples in the form (id, avg distances, children).
    If a node is a leaf, distances is just set to 0 and children to [].
    """
    for n in level:
        if h.g.is_cluster(n):
            distances = np.mean(h.get_nearest_distances(n))
            children = [ch for ch in h.g.get_children(n)]
            yield (n, distances, [ch for ch in _build_tree(h, children)])
        else:
            yield (n, 0, [])
