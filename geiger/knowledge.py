"""
These data sources take extremely long to load,
so this is an abstraction which allows the running
of them in a separate process so they don't constantly
need to be reloaded.

Calling out to a separate process is slower, but
cuts down on loading time.

The interface is common whether it's a separate process or not.
"""

from gensim.models.word2vec import Word2Vec
from gensim.models import Phrases
from multiprocessing.connection import Client


_w2v = None
_phrases = None


class Bigram():
    def __init__(self, remote):
        global _phrases
        self.remote = remote
        if not remote and _phrases is None:
            print('Loading phrases model...')

            # Trained on 100-200k NYT articles
            _phrases = Phrases.load('data/bigram_model.phrases')
        else:
            address = ('localhost', 6001)
            self.conn = Client(address, authkey=b'password')

    def __getitem__(self, word):
        if self.remote:
            self.conn.send(word)
            return self.conn.recv()
        else:
            return _phrases[word]


class W2V():
    def __init__(self, remote):
        global _w2v
        self.remote = remote
        if not remote and _w2v is None:
            print('Loading word2vec model...')
            _w2v = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
            self.vocab = Vocab(remote, None)
        else:
            address = ('localhost', 6000)
            self.conn = Client(address, authkey=b'password')
            self.vocab = Vocab(remote, self.conn)

    def similarity(self, t1, t2):
        if self.remote:
            self.conn.send((t1,t2))
            return self.conn.recv()
        else:
            return _w2v.similarity(t1, t2)

    def n_similarity(self, t1, t2):
        if self.remote:
            self.conn.send((t1,t2))
            return self.conn.recv()
        else:
            return _w2v.n_similarity(t1, t2)


class Vocab():
    def __init__(self, remote, conn):
        self.remote = remote
        self.conn = conn

    def __contains__(self, word):
        if self.remote:
            self.conn.send(('vocab', word))
            return self.conn.recv()
        else:
            return word in _w2v.vocab
