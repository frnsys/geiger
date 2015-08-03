import numpy as np
from gensim.matutils import Scipy2Corpus
from gensim.models.ldamulticore import LdaMulticore
from broca import Pipe


class LDA(Pipe):
    """
    LDA (Latent Dirichlet Allocation) model
    for unsupervised topic modeling.

    Takes vectors and returns topic vectors,
    which can be used for clustering.
    """
    input = Pipe.type.vecs
    output = Pipe.type.vecs

    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.trained = False

    def __call__(self, vecs):
        """
        Return topic vectors.
        """
        if not self.trained:
            self.train(vecs)
            self.trained = True

        distribs = []
        for distrib in self.m[Scipy2Corpus(vecs)]:
            distribs.append([t[1] for t in distrib])
        distribs = np.array(distribs)
        return distribs

    def train(self, vecs):
        """
        Build the topic model.
        """
        corp = Scipy2Corpus(vecs)
        self.m = LdaMulticore(corp, num_topics=self.n_topics, iterations=1000, workers=3)

    def print_topics(self, vectorizer):
        vocab = vectorizer.vocabulary
        for topic in self.m.show_topics(num_topics=self.n_topics, num_words=10, formatted=False):
            print([vocab[int(ix)] for prob, ix in topic])
