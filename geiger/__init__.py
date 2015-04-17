import random
from nltk.tokenize import sent_tokenize
from geiger import clustering, sentences
from geiger.featurizers import Featurizer

"""
These all return results in the format:

    [(sentence body, sentence comment, support, cohort), ...]

"""

def lda_extract_by_topics(comments, n_topics=5):
    clusters, lda = clustering.lda(comments, n_topics=n_topics)
    return sentences.extract_by_topics(clusters, lda)


def lda_extract_by_distance(comments, n_topics=5):
    # Build features for comments for later use.
    f = Featurizer()
    f.featurize(comments)
    clusters, lda = clustering.lda(comments, n_topics=n_topics)
    return sentences.extract_by_distance(clusters, f)


def kmeans_extract_by_distance(comments):
    f = Featurizer()
    clusters = clustering.k_means(comments, f)
    return sentences.extract_by_distance(clusters, f)


def hac_extract_by_distance(comments):
    f = Featurizer()
    clusters = clustering.hac(comments, f)
    return sentences.extract_by_distance(clusters, f)


def dbscan_extract_by_distance(comments):
    f = Featurizer()
    clusters = clustering.dbscan(comments, f)
    return sentences.extract_by_distance(clusters, f)


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
