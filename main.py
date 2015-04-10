import sys
import pandas as pd
from sklearn.externals import joblib

import config
from server import app

from geiger.models.doc2vec import Model as Doc2Vec


def train_doc2vec():
    """
    Train the doc2vec model.
    """
    m = Doc2Vec()
    m.train('data/commentBodies.txt')


def server():
    """
    Run the demo server.
    """
    app.run(debug=True, port=5001)


def comments():
    """
    Download up to the first 300 comments for a given NYT article url.
    """
    from geiger.services import get_comments
    url = sys.argv[2]

    comments = get_comments(url, n=300)

    texts = [c.body for c in comments]

    with open('out.txt', 'w') as f:
        f.write('\n\n\n\n\n'.join(texts))


def eval():
    """
    Try clustering on hand-clustered data.

    This is probably super unreliable since there's so little data, but it's a starting point :)
    """
    import json
    path_to_examples = 'data/examples.json'
    clusters = json.load(open(path_to_examples, 'r'))

    class Doc():
        def __init__(self, doc):
            self.body = doc

    docs = []
    labels = []
    for i, clus in enumerate(clusters):
        for doc in clus:
            docs.append(Doc(doc))
            labels.append(i)

    from geiger.util.evaluate import evaluate
    print(evaluate(docs, labels))


if __name__ == '__main__':
    # Convenient, but hacky
    globals()[sys.argv[1]]()