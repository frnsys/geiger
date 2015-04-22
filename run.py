import sys
import json
from server import app

from geiger.comment import Doc
from geiger.util.evaluate import evaluate
from geiger.models.doc2vec import Model as Doc2Vec
from geiger.models.polisent import Model as Polisent


def train_doc2vec():
    """
    Train the doc2vec model.
    """
    m = Doc2Vec()
    m.train([
        'data/all_approved_comments_01.csv',
        'data/all_approved_comments_02.csv',
        'data/all_approved_comments_03.csv'
    ])


def train_polisent():
    print('Training the polisent model...')
    m = Polisent()
    m.train()
    print('Done.')


def server():
    """
    Run the demo server.
    """
    app.run(host='0.0.0.0', debug=True, port=5001)


def eval():
    """
    Try clustering on hand-clustered data.

    This is probably super unreliable since there's so little data, but it's a starting point :)
    """
    path_to_examples = 'data/examples.json'
    clusters = json.load(open(path_to_examples, 'r'))

    docs = []
    labels = []
    for i, clus in enumerate(clusters):
        for doc in clus:
            docs.append(Doc(doc))
            labels.append(i)

    results = evaluate(docs, labels)
    print(json.dumps(results, indent=4, sort_keys=True))


if __name__ == '__main__':
    # Convenient, but hacky
    globals()[sys.argv[1]]()