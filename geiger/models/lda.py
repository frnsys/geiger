import lda
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import sent_tokenize
from geiger.text import strip_tags, Tokenizer
import logging


# Silence handlers from the `lda` library,
# this is probably overkill :\
logger = logging.getLogger('lda')
logger.handlers = []

class Model():
    """
    LDA (Latent Dirichlet Allocation) model
    for unsupervised topic modeling.

    TO DO:
        - this model has to be rebuilt for each comment section as new comments come in - what's the best way to manage that?

    Notes:
        - tried LDA on individual sentences, doesn't work as well.
    """

    def __init__(self, n_topics=None, verbose=False):
        """
        Args:
            | topics            -- int or None, number of topics. If None, will try a range to maximize the log likelihood.
        """
        self.verbose = verbose
        self.n_topics = n_topics
        self.vectr = CountVectorizer(stop_words='english', ngram_range=(1,1), tokenizer=Tokenizer())

    def train(self, docs):
        """
        Build the topic model from a list of documents (strings).

        Assumes documents have been pre-processed (e.g. stripped of HTML, etc)
        """
        vecs = self.vectr.fit_transform(docs)

        # If n_topics is not specified, try a range.
        if self.n_topics is None:
            # Try n_topics in [5, 20] in steps of 2.
            n_topics_range = np.arange(5, 20, 2)

            results = []
            models = []
            for n in n_topics_range:
                model = lda.LDA(n_topics=n, n_iter=2000, random_state=1)
                model.fit_transform(vecs)
                models.append(model)
                results.append(model.loglikelihood())

                if self.verbose:
                    self.print_topics(model)

            i = np.argmax(results)
            self.n_topics = n_topics_range[i]
            self.m = models[i]

        else:
            self.m = lda.LDA(n_topics=self.n_topics, n_iter=2000, random_state=1)
            self.m.fit_transform(vecs)
            if self.verbose:
                self.print_topics(self.m)

    @property
    def topic_dists(self):
        return self.m.doc_topic_

    def cluster(self, docs):
        """
        Build clusters out of most likely topics.
        """

        # If no model exists, train it.
        if not hasattr(self, 'm'):
            self.train(docs)

        clusters = [[] for i in range(self.n_topics)]
        for i, doc in enumerate(docs):
            topic = self.m.doc_topic_[i].argmax()
            clusters[topic].append(doc)

        return clusters

    def print_topics(self, model):
        """
        Prints out the top words for each topic in the model.
        """
        n_top_words = 8
        topic_word = model.components_
        vocab = self.vectr.get_feature_names()
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
            print('Topic: {}: {}'.format(i, ' | '.join(topic_words)))
        print('---')
