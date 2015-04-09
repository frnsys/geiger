from geiger.text import strip_tags
from geiger.models import lda

class Featurizer():
    """
    Topics featurizer.
    This uses LDA to generate a n-dimension feature vector of topic probabilities,
    where n is the number of topics.
    """

    def __init__(self, n_topics=None):
        self.m = lda.Model(n_topics=n_topics)

    def featurize(self, comments, return_ctx=False):
        self.m.train([strip_tags(c.body) for c in comments])
        topic_dists = self.m.topic_dists

        if return_ctx:
            return topic_dists, topic_dists
        else:
            return topic_dists
