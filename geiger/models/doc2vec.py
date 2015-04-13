import os
import string
import config
from time import time
from nltk import word_tokenize
from nltk.corpus import stopwords
from geiger.util.progress import Progress
from gensim.models.doc2vec import Doc2Vec, LabeledSentence


class Model():
    """
    This is an augmentation of Word2Vec which is capable of creating vector
    representations for larger texts (i.e. beyond individual words).
    These vectors are of consistent dimension which makes it a potential replacement
    for the traditional bag-of-words representation.

    See:

        - <http://radimrehurek.com/gensim/models/doc2vec.html>
        - <http://radimrehurek.com/2014/12/doc2vec-tutorial/>
        - Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. <http://arxiv.org/pdf/1405.4053v2.pdf>
    """

    def __init__(self):
        self.path = config.d2v_path
        if os.path.exists(self.path):
            self.m = Doc2Vec.load(self.path)

            # We leave word representations fixed, henceforth only learn representations of documents.
            # Or is it worthwhile to continue updating word representations?
            self.m.train_words = False

    def train(self, data_paths):
        """
        Training
        ~~~~~~~~

        Train the `Doc2Vec` model.

        Since we don't care about the vector representations of the training docs,
        we only need to train the word representations (`train_lbls=False`)

        Some documentation at <http://radimrehurek.com/2014/12/doc2vec-tutorial/> suggests:

            ...iterating over the data several times and either
                1. randomizing the order of input sentences, or
                2. manually controlling the learning rate over the course of several iterations.

            For example, if one wanted to manually control the learning rate over the course of 10 epochs, one could use the following:

                model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
                model.build_vocab(sentences)

                for epoch in range(10):
                    model.train(sentences)
                    model.alpha -= 0.002  # decrease the learning rate
                    model.min_alpha = model.alpha  # fix the learning rate, no decay

        On ~1.6 million comments, training took ~19min.
        """


        # For now, use mostly default settings.
        print('Training model...')
        t0 = time()
        self.m = Doc2Vec(
                sentences=self._doc_stream(data_paths),
                train_lbls=False,       # only train word representations
                workers=5,              # multithreading
                min_count=50,           # ignore words with frequency less than this
                size=300,               # dimensionality of feature vectors
                sample=0.001,           # threshold for which higher-frequency words are randomly downsampled
                window=10,              # amount of context to use (dist b/w current and predicted words)
            )
        print("Trained in {0:.2f}s".format(time() - t0))

        print('Saving model...')
        self.m.save(self.path)
        print('Done.')

        # We leave word representations fixed, henceforth only learn representations of documents.
        # Or is it worthwhile to continue updating word representations?
        self.m.train_words = False


    punct_map = {ord(p): ' ' for p in string.punctuation + '“”'}
    period_map = {ord('.'): None} # To preserve initialisms, e.g. F.D.A. -> FDA
    def _tokenize(self, doc):
        """
        For tokenization, we do not want to remove
        stopwords or stem/lemmatize words. The `Doc2Vec` model learns
        from the context of words, so we need to preserve it.

        Though we do remove/clean up some punctuation.
        """

        doc = doc.translate(self.period_map)
        doc = doc.translate(self.punct_map)
        return word_tokenize(doc.lower())


    def _doc_stream(self, paths):
        """
        A generator to return processed documents (comments).
        """
        p = Progress('DOC2VEC')

        n = 0
        for path in paths:
            n += sum(1 for line in open(path, 'r'))
        print('Using {0} documents.'.format(n))

        i = 0
        for path in paths:
            with open(path, 'r') as f:
                for line in f:
                    i += 1
                    p.print_progress(i/n)
                    tokens = self._tokenize(line)
                    yield LabeledSentence(tokens, labels=['DOC_{0}'.format(i)])


    def infer_vector(self, doc):
        """
        Featurizing
        ~~~~~~~~~~~

        Build feature vectors for documents using the `Doc2Vec` model.

        The inference function which generates vector representations of new documents
        is not yet implemented in the main `gensim` package (as of 04.10.2015);
        a development version is available in this fork: <https://github.com/gojomo/gensim>

            $ pip install git+https://github.com/gojomo/gensim@develop

        The development implementation of `infer_vector(document)` has not yet been optimized
        so it won't be very fast for bulk processing. It should be ok for development.

        """
        tokens = self._tokenize(doc)
        return self.m.infer_vector(tokens)


    stops = stopwords.words('english')
    def _infer_tokenize(self, doc):
        """
        not sure if this helps
        """
        doc = doc.translate(self.period_map)
        doc = doc.translate(self.punct_map)
        return [t for t in word_tokenize(doc) if t not in self.stops]
