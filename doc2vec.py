"""
Run the Doc2Vec model as a separate process,
that way it doesn't need to keep being reloaded.
"""

# Listener
from multiprocessing.connection import Listener
from geiger.models.doc2vec import Model as Doc2Vec

print('Loading Doc2Vec model...')
m = Doc2Vec()

print('Creating listener...')
address = ('localhost', 6000)
with Listener(address, authkey=b'password') as listener:
    while True:
        with listener.accept() as conn:
            print('connection accepted from {0}'.format(listener.last_accepted))
            while True:
                try:
                    msg = conn.recv()
                    conn.send(m.infer_vector(msg))
                except (EOFError, ConnectionResetError):
                    break
