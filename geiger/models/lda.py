import numpy as np
from geiger.text import Vectorizer
from geiger.util.progress import Progress
from gensim.matutils import Scipy2Corpus
from gensim.models.ldamulticore import LdaMulticore


class Model():
    """
    LDA (Latent Dirichlet Allocation) model
    for unsupervised topic modeling.

    TO DO:
        - this model has to be rebuilt for each comment section as new comments come in - what's the best way to manage that?

    Notes:
        - tried LDA on individual sentences, doesn't work as well.
    """

    def __init__(self, n_topics=5, verbose=False):
        self.verbose = verbose
        self.n_topics = n_topics
        self.vectr = Vectorizer()

    def train(self, comments):
        """
        Build the topic model from a list of documents (strings).

        Assumes documents have been pre-processed (e.g. stripped of HTML, etc)
        """
        docs = [c.body for c in comments]
        vecs = self.vectr.vectorize(docs, train=True)
        corp = Scipy2Corpus(vecs)
        self.m = LdaMulticore(corp, num_topics=self.n_topics, iterations=1000, workers=3)

        if self.verbose:
            self.print_topics()

    def featurize(self, docs):
        """
        Return topic vectors for documents.
        """
        vecs = self.vectr.vectorize(docs)

        dists = []
        for dist in self.m[Scipy2Corpus(vecs)]:
            dists.append([t[1] for t in dist])
        dists = np.array(dists)
        return dists

    def cluster(self, comments):
        """
        Build clusters out of most likely topics.
        """

        # If no model exists, train it.
        if not hasattr(self, 'm'):
            self.train(comments)

        clusters = [[] for _ in range(self.n_topics)]
        dists = self.featurize(comments)
        for i, comment in enumerate(comments):
            topic = dists[i].argmax()
            clusters[topic].append(comment)

        return clusters

    def print_topics(self):
        vocab = self.vectr.vocabulary
        for topic in self.m.show_topics(num_topics=self.n_topics, num_words=10, formatted=False):
            print([vocab[int(ix)] for prob, ix in topic])
