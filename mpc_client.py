from multiprocessing.connection import Client

address = ('localhost', 6000)
with Client(address, authkey=b'password') as conn:
    while True:
        conn.send('this is a test document')
        msg = conn.recv()
        print(msg)
