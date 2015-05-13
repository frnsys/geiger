"""
Run some slow-loading things as separate process
that way they don't need to keep being reloaded.
"""

from multiprocessing.connection import Listener
from gensim.models.word2vec import Word2Vec
from gensim.models import Phrases

def word2vec():
    print('Loading word2vec model...')
    w2v = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

    print('Creating listener...')
    address = ('localhost', 6000)
    with Listener(address, authkey=b'password') as listener:
        while True:
            with listener.accept() as conn:
                print('connection accepted from {0}'.format(listener.last_accepted))
                while True:
                    try:
                        msg = conn.recv()
                        try:
                            if msg[0] == 'vocab':
                                conn.send(msg[1] in w2v.vocab)
                            elif isinstance(msg[0], list):
                                conn.send(w2v.n_similarity(*msg))
                            else:
                                conn.send(w2v.similarity(*msg))
                        except KeyError:
                            conn.send(0.)
                    except (EOFError, ConnectionResetError):
                        break


def phrases():
    print('Loading phrases model...')
    bigram = Phrases.load('data/bigram_model.phrases')

    print('Creating listener...')
    address = ('localhost', 6001)
    with Listener(address, authkey=b'password') as listener:
        while True:
            with listener.accept() as conn:
                print('connection accepted from {0}'.format(listener.last_accepted))
                while True:
                    try:
                        msg = conn.recv()
                        conn.send(bigram[msg])
                    except (EOFError, ConnectionResetError):
                        break


if __name__ == '__main__':
    # Convenient, but hacky
    globals()[sys.argv[1]]()
