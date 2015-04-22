from multiprocessing.connection import Client

address = ('localhost', 6000)
with Client(address, authkey=b'password') as conn:
    while True:
        conn.send(('document', 'paper'))
        msg = conn.recv()
        print(msg)
