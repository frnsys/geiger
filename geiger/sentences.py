import numpy as np
from nltk import sent_tokenize
from scipy.spatial.distance import cdist
from geiger.featurizers import featurize

def select_sentences(clusters):
    """
    Extract representative sentences from a list of Comment clusters.

    Current strategy:

        - for each cluster
            - calculate centroid (mean)
            - for each comment
                - tokenize into sentences
                - for each sentence
                    - calculate relevancy to parent cluster
                        - few diff options
                            - aspect-based
                                - identify aspects (noun phrases or keywords)
                                - calculate support for each aspect
                                    (i.e. what fraction of other sentences mention it?)
                            - topic-based (if using LDA for clustering)
                                - use topic model for comments and identify topic for sentence
                                - relevancy = probability of assignment to parent cluster topic
                            - distance (this may work well with doc2vec)
                                - featurize sentence
                                - calculate distance to centroid
            - select sentence with greatest relevancy

    Can control for other factors too:

        - skip sentences with a non-"I" pronoun
        - skip sentences below some threshold length
    """
    #for clus in clusters:
        #np.mean
        #for comment in clus:
            #for sent in sent_toknize(comment.body):

class Sentence():
    def __init__(self, body):
        self.body = body


def extract_by_distance(clusters, top_n=5):
    """
    For each cluster, calculate a centroid vector (mean of its children's feature vectors),
    then select the sentence closest to the centroid.

    TO DO not working
    """
    for clus in clusters:
        agg_feats = np.vstack([c.features for c in clus])
        print(agg_feats.shape)
        centroid = np.array([ np.mean(agg_feats, axis=0) ])
        print(centroid.shape)
        for comment in clus:
            feats = featurize([Sentence(sent) for sent in sent_tokenize(comment.body)])
            print(feats.shape)
            dists = cdist(centroid, feats, metric='cosine')
            print(dists)


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


