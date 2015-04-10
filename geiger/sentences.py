import numpy as np
from nltk import sent_tokenize
from scipy.spatial.distance import cdist
from geiger.featurizers import featurize


class Sentence():
    def __init__(self, body):
        self.body = body


def extract_by_distance(clusters, top_n=5):
    """
    For each cluster, calculate a centroid vector (mean of its children's feature vectors),
    then select the sentence closest to the centroid.
    """
    results = []
    for clus in clusters:
        # Calculate centroid for the cluster.
        agg_feats = np.vstack([c.features for c in clus])
        centroid = np.array([ np.mean(agg_feats, axis=0) ])

        # Calculate features for each sentence in the cluster.
        feats = []
        sents = []
        for comment in clus:
            sents += [Sentence(sent) for sent in sent_tokenize(comment.body) if len(sent) >= 10]
        feats = featurize(sents)

        # Calculate distances to the centroid.
        dists = cdist(centroid, feats, metric='cosine')

        # Select the closest sentence.
        i = np.argmin(dists)
        results.append(sents[i].body)

    sizes = np.array([len(clus) for clus in clusters])
    max_idx = np.argpartition(sizes, -top_n)[-top_n:]

    return [(results[i], sizes[i]) for i in max_idx]


def extract_by_topics(clusters, lda, top_n=5):
    """
    Use the topic model trained on the comments
    and identify a topic for each sentence.

    For each cluster, select sentences which are of the same topic
    as the cluster.

    Then pick the sentence assigned that topic with the highest probability.

    *NOTE* This only works if clustering was done by LDA. It is assumed that the clusters
    are in order of topics (that is, the index of the cluster in `clusters` is its topic number.)

    Relevance = probability of assignment to the parent cluster topic

    Args:
        | clusters      -- list of clusters
        | lda           -- the LDA model built on the
                            comments in the clusters
    """
    results = []
    for topic, clus in enumerate(clusters):
        clus_sents = []
        for comment in clus:
            sents = sent_tokenize(comment.body)

            # Filter by length
            sents = [sent for sent in sents if len(sent) > 100]

            # Select only sentences which are congruous with the parent topic.
            clus_sents += [(sent, prob) for sent, sent_topic, prob in lda.identify(sents) if sent_topic == topic]

        # Select the relevant sentence with the highest probability.
        rep_sent = max(clus_sents, key=lambda c: c[1])
        results.append(rep_sent[0])

    # TO DO this can be pulled up so we don't process all clusters.
    # Select the indices of the top 5 largest clusters.
    sizes = np.array([len(clus) for clus in clusters])
    max_idx = np.argpartition(sizes, -top_n)[-top_n:]

    return [(results[i], sizes[i]) for i in max_idx]


