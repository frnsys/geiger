import random
import numpy as np
from nltk import sent_tokenize, word_tokenize
from scipy.spatial.distance import cdist
from geiger.aspects import extract_aspects
from geiger.aspects.apriori import apriori


class Sentence():
    def __init__(self, body, comment):
        self.body = body
        self.comment = comment


def prefilter(sentence):
    """
    Ignore sentences for which this returns False.
    """
    tokens = word_tokenize(sentence.lower())
    first_word = tokens[0]
    first_char = first_word[0]
    final_char = tokens[-1][-1]

    # Filter out short sentences.
    if len(tokens) < 10:
        return False

    # The following rules are meant to filter out sentences
    # which may require extra context.
    elif first_char in ['"', '(', '\'', '*', '“', '‘', ':']:
        return False
    elif first_word in ['however', 'so', 'for', 'or', 'and', 'thus', 'therefore', 'also', 'firstly', 'secondly', 'thirdly']:
        return False
    elif set(tokens).intersection({'he', 'she', 'it', 'they', 'them', 'him', 'her', 'their', 'I'}):
        return False
    elif final_char in ['"', '”', '’']:
        return False

    return True


def extract_by_distance(clusters, featurizer, top_n=5):
    """
    For each cluster, calculate a centroid vector (mean of its children's feature vectors),
    then select the sentence closest to the centroid.

    Returns a list of tuples:

        [(sentence body, sentence comment, cluster size, cohort), ...]

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
            sents += [Sentence(sent, comment) for sent in sent_tokenize(comment.body) if prefilter(sent)]
        feats = featurizer.featurize(sents)

        # Calculate distances to the centroid.
        dists = cdist(centroid, feats, metric='cosine')

        # Select the closest sentence.
        i = np.argmin(dists)
        results.append(sents[i])

    sizes = np.array([len(clus) for clus in clusters])
    max_idx = np.argpartition(sizes, -top_n)[-top_n:]

    return [(results[j].body, results[j].comment, sizes[j], clusters[j]) for j in max_idx]


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

    Returns a list of tuples:

        [(sentence body, sentence comment, cluster size, cohort), ...]

    """
    results = []
    for topic, clus in enumerate(clusters):
        clus_sents = []
        for comment in clus:
            sents = sent_tokenize(comment.body)

            # Filter by length
            sents = [sent for sent in sents if prefilter(sent)]

            # Select only sentences which are congruous with the parent topic.
            clus_sents += [(Sentence(sent, comment), prob) for sent, sent_topic, prob in lda.identify(sents) if sent_topic == topic]

        # Select the relevant sentence with the highest probability.
        rep_sent = max(clus_sents, key=lambda c: c[1])
        results.append(rep_sent[0])

    # TO DO this can be pulled up so we don't process all clusters.
    # Select the indices of the top 5 largest clusters.
    sizes = np.array([len(clus) for clus in clusters])
    max_idx = np.argpartition(sizes, -top_n)[-top_n:]

    return [(results[i].body, results[i].comment, sizes[i], clusters[i]) for i in max_idx]


def extract_by_aspects(comments, strategy='pos_tag'):
    """
    Takes all comments, tries to identify the most commonly-discussed
    aspects, and picks a random representative for each.

    Note that the input here is not a list of clusters; rather it is
    just a list of comments.

    Returns a list of tuples:

        [(sentence body, sentence comment, support, cohort), ...]

    Note: here the support value is not the # of comments, but the # of sentences.
    and the cohort consists of sentences, not comments.

    TO DO don't pick sentences randomly, rank them in some way.
    """
    sents = []
    for comment in comments:
        sents += [Sentence(sent, comment) for sent in sent_tokenize(comment.body) if prefilter(sent)]

    # Calculate support for each aspect.
    counts = {}
    for sent in sents:
        sent.aspects = extract_aspects(sent.body)
        for aspect in sent.aspects:
            if aspect not in counts:
                counts[aspect] = 0
            counts[aspect] += 1

    # Sort and get top n aspects.
    count_sorted = sorted(counts.items(), key=lambda c: c[1], reverse=True)
    top_aspects = [k[0] for k in count_sorted[:5]]

    # Find sentences for each aspect.
    aspects = {k: [] for k in top_aspects}
    for sent in sents:
        overlap = set(sent.aspects).intersection(top_aspects)
        for aspect in overlap:
            aspects[aspect].append(sent)

    # Pick a random sentence for each aspect.
    results = []
    for aspect, sents in aspects.items():
        sent = random.choice(sents)
        results.append((sent.body, sent.comment, counts[aspect], sents))
    return results


from geiger.aspects import extract_aspect_candidates
def extract_by_apriori(comments, min_sup=0.05):
    """
    This is similar to `extract_by_aspects` but uses the Apriori algorithm.

    Returns a list of tuples:

        [(sentence body, sentence comment, support, cohort), ...]

    Note: here the support value is not the # of comments, but the # of sentences,
    and the cohort consists of sentences, not comments.

    TO DO tweak `min_sup` param, for small amounts of comments (<=100), may be hard
    to identify proper aspects.

    TO DO don't pick sentences randomly, rank them in some way.

    TO DO Aspect identification might be better if we lemmatize them.
    But then need to keep track of which lemmas map to which sentences (can't check via `in`).
    """
    sents = []
    for comment in comments:
        sents += [Sentence(sent, comment) for sent in sent_tokenize(comment.body) if prefilter(sent)]

    aspects = apriori([extract_aspect_candidates(s.body.lower()) for s in sents], min_sup=min_sup)

    # Cluster based on aspects.
    # This could be cleaned up/made more efficient
    aspect_sents = {k: [] for k in aspects}
    counts = {k: 0 for k in aspects}
    for sent in sents:
        for aspect in aspects:
            if aspect in sent.body.lower():
                counts[aspect] += 1
                aspect_sents[aspect].append(sent)

    # Pick a random sentence for each aspect.
    results = []
    for aspect, sents in aspect_sents.items():
        sent = random.choice(sents)
        results.append((sent.body, sent.comment, counts[aspect], sents))
    return results
