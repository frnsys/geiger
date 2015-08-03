"""
These data sources take extremely long to load,
so this is an abstraction which allows the running
of them in a separate process so they don't constantly
need to be reloaded.

Calling out to a separate process is slower, but
cuts down on loading time.

The interface is common whether it's a separate process or not.
"""

import json
from gensim.models.word2vec import Word2Vec
from gensim.models import Phrases
from multiprocessing.connection import Client


_w2v = None
_phrases = None
_idf = None

_w2v_conn = None
_phrases_conn = None
_idf_conn = None

class Bigram():
    def __init__(self, remote):
        global _phrases
        global _phrases_conn

        self.remote = remote
        if not remote and _phrases is None:
            print('Loading phrases model...')

            # Trained on 100-200k NYT articles
            _phrases = Phrases.load('data/nyt/bigram_model.phrases')
            print('Done loading phrases')
        elif _phrases_conn is None:
            print('Connecting to phrases process...')
            address = ('localhost', 6001)
            _phrases_conn = Client(address, authkey=b'password')
            print('Done connecting to phrases')
        self.conn = _phrases_conn

    def __getitem__(self, word):
        if self.remote:
            self.conn.send(word)
            return self.conn.recv()
        else:
            return _phrases[word]


class IDF():
    def __init__(self, remote):
        global _idf
        global _idf_conn

        self.remote = remote
        if not remote and _idf is None:
            print('Loading idf...')
            _idf = json.load(open('data/nyt/idf.json', 'r'))

            # Normalize
            mxm = max(_idf.values())
            for k, v in _idf.items():
                _idf[k] = v/mxm
            print('Done loading idf')

        elif _idf_conn is None:
            print('Connecting to idf process...')
            address = ('localhost', 6002)
            _idf_conn = Client(address, authkey=b'password')
            print('Done connecting to idf')
        self.conn = _idf_conn

    def __getitem__(self, term):
        if self.remote:
            self.conn.send(term)
            return self.conn.recv()
        else:
            return _idf[term]

    def get(self, term, default):
        if self.remote:
            self.conn.send(term)
            return self.conn.recv()
        else:
            return _idf.get(term, default)


class W2V():
    def __init__(self, remote):
        global _w2v
        global _w2v_conn

        self.remote = remote
        if not remote and _w2v is None:
            print('Loading word2vec model...')
            _w2v = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
            self.vocab = Vocab(remote, None)
            print('Done loading word2vec')
        elif _w2v_conn is None:
            print('Connecting to word2vec process...')
            address = ('localhost', 6000)
            _w2v_conn = Client(address, authkey=b'password')
            self.vocab = Vocab(remote, _w2v_conn)
            print('Done connecting to word2vec')
        self.conn = _w2v_conn

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
