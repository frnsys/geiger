"""
Run the Doc2Vec model as a separate process,
that way it doesn't need to keep being reloaded.
"""

# Listener
from multiprocessing.connection import Listener
from gensim.models import Phrases

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