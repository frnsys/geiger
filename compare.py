"""
Compare the outputs of different strategies.
"""

import sys
import json
from geiger import clustering, sentences, aspects, services
from geiger.featurizers import featurize
from geiger.text import strip_tags

class FauxComment():
    def __init__(self, body):
        self.body = strip_tags(body)

def fanfare(text):
    """
    trumpets & confetti go here
    """
    print('🎉 '*32)
    print('🎉 {0}'.format(text))
    print('🎉 '*32)


def lda_extract_by_topics(comments):
    # LDA + extract_by_topics
    print('Training topic model...')
    clusters, lda = clustering.lda(comments, n_topics=None)
    return sentences.extract_by_topics(clusters, lda)

def lda_extract_by_distance(comments):
    # Build features for comments for later use.
    featurize(comments)
    clusters, lda = clustering.lda(comments, n_topics=13)
    return sentences.extract_by_distance(clusters)

def kmeans_extract_by_distance(comments):
    clusters = clustering.k_means(comments)
    return sentences.extract_by_distance(clusters)

def aspects_only_pos(comments):
    return aspects.summarize_aspects(comments, strategy='pos_tag')

def aspects_only_rake(comments):
    return aspects.summarize_aspects(comments, strategy='rake')

def compare(url):
    # Load comments.
    if url is not None:
        print('Fetching comments...')
        comments = services.get_comments(url, n=300)
        comments = [c for c in comments if len(c.body) > 140]

    else:
        print('Loading example comments...')
        path_to_examples = 'data/examples.json'
        clusters = json.load(open(path_to_examples, 'r'))
        docs = []
        for clus in clusters:
            for doc in clus:
                docs.append(doc)
        comments = [FauxComment(d) for d in docs if len(doc) > 140]

    print('Using {0} comments.'.format(len(comments)))

    # Try out different approaches.
    for func in [
            kmeans_extract_by_distance,
            lda_extract_by_distance,
            lda_extract_by_topics,
            aspects_only_pos,
            aspects_only_rake
        ]:
        fanfare(func.__name__)
        yield func.__name__, func(comments)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = None

    for strat, results in compare(url):
        print(results)
