import sys
import config
import pandas as pd
from server import app
from geiger.text import Vectorizer, strip_tags, html_decode
from sklearn.externals import joblib


def train():
    """
    Train the vectorizer.
    """
    v = Vectorizer()

    print('Loading data...')
    data = pd.read_csv(config.comments_path, index_col=0, lineterminator='\n')

    print('Training the vectorizer on {0} comments...'.format(len(data)))
    comments = [strip_tags(html_decode(c)) for c in data['commentBody'].to_dict().values()]
    v.vectorize(comments, train=True)
    joblib.dump(v, config.vectorizer_path)


def server():
    """
    Run the demo server.
    """
    app.run(debug=True, port=5001)


if __name__ == '__main__':
    # Convenient, but hacky
    globals()[sys.argv[1]]()