import config
from server import app

if __name__ == '__main__':
    # Something about multiprocesses conflicts with the reloader
    app.run(host='0.0.0.0', debug=True, port=5001, use_reloader=not config.remote)