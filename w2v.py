"""
Run the Doc2Vec model as a separate process,
that way it doesn't need to keep being reloaded.
"""

# Listener
from multiprocessing.connection import Listener
from gensim.models.word2vec import Word2Vec

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
                        conn.send(w2v.similarity(*msg))
                    except KeyError:
                        conn.send(0.)
                except (EOFError, ConnectionResetError):
                    break
