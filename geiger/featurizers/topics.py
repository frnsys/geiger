from geiger.models import lda

class Featurizer():
    """
    Topics featurizer.
    This uses LDA to generate a n-dimension feature vector of topic probabilities,
    where n is the number of topics.
    """

    def __init__(self, n_topics=None):
        self.trained = False
        self.m = lda.Model(n_topics=n_topics)

    def featurize(self, comments, return_ctx=False):
        if not self.trained:
            self.m.train(comments)
            topic_dists = self.m.topic_dists
        else:
            topic_dists = self.m.featurize([c.body for c in comments])

        self.trained = True
        if return_ctx:
            return topic_dists, topic_dists
        else:
            return topic_dists
