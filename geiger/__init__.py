import random
from nltk.tokenize import sent_tokenize
from geiger import clustering, sentences, aspects
from geiger.featurizers import featurize

"""
These all return results in the format:

    [(sentence body, sentence comment, support), ...]

"""

def lda_extract_by_topics(comments, n_topics=None):
    clusters, lda = clustering.lda(comments, n_topics=n_topics)
    return sentences.extract_by_topics(clusters, lda)


def lda_extract_by_distance(comments, n_topics=None):
    # Build features for comments for later use.
    featurize(comments)
    clusters, lda = clustering.lda(comments, n_topics=n_topics)
    return sentences.extract_by_distance(clusters)


def kmeans_extract_by_distance(comments):
    clusters = clustering.k_means(comments)
    return sentences.extract_by_distance(clusters)


def aspects_only_pos(comments):
    return sentences.extract_by_aspects(comments, strategy='pos_tag')


def aspects_only_rake(comments):
    return sentences.extract_by_aspects(comments, strategy='rake')


def aspects_only_apriori(comments):
    return sentences.extract_by_apriori(comments)


def baseline(comments):
    """
    Baseline: select 5 random sentences.
    """
    sents = []
    for c in comments:
        sents += [(sent, c) for sent in sent_tokenize(c.body)]

    results = []
    for i in range(5):
        sel = random.choice(sents)
        results.append((sel[0], sel[1], 0, []))
        sents.remove(sel)
    return results
