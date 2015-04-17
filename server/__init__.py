import json
import config
import geiger
from geiger.text import strip_tags
from geiger.comment import Comment
from geiger import services, clustering
from geiger.featurizers import Featurizer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/')
def index():
    strats = [
        'kmeans_extract_by_distance',
        'hac_extract_by_distance',
        'dbscan_extract_by_distance',
        'lda_extract_by_distance',
        'lda_extract_by_topics',
        'aspects_only_pos',
        'aspects_only_rake',
        'aspects_only_apriori',
        'baseline'
    ]

    resolution = 'sentences' if config.sentences else 'comments'
    return render_template('index.html', strategies=strats, featurizers=config.featurizers, resolution=resolution)


@app.route('/visualize/<strategy>/', defaults={'url':''})
@app.route('/visualize/<strategy>/<path:url>')
def visualize(strategy, url):
    """
    For more closely examining different clustering strategies.
    """

    if url:
        comments = services.get_comments(url, n=300)
        comments = [c for c in comments if len(c.body) > 140]

    else:
        path_to_examples = 'data/examples.json'
        clusters = json.load(open(path_to_examples, 'r'))
        docs = []
        for clus in clusters:
            for doc in clus:
                docs.append(doc)

        docs = [strip_tags(doc) for doc in docs if len(doc) >= 140] # drop short comments :D

        class FauxComment():
            def __init__(self, body):
                self.body_html = body
                self.body = strip_tags(body)

        comments = [FauxComment(d) for d in docs]

    strats = {
        'lda': clustering.lda,
        'hac': clustering.hac,
        'ihac': clustering.ihac,
        'k_means': clustering.k_means,
        'dbscan': clustering.dbscan
    }

    if strategy not in strats:
        return 'No clustering strategy by the name "{0}". Use one of {1}.'.format(strategy, list(strats.keys())), 404
    else:
        if strategy == 'lda':
            clusters = strats[strategy](comments, return_ctx=True)
        else:
            f = Featurizer()
            clusters = strats[strategy](comments, f, return_ctx=True)

    return render_template('visualize.html', clusters=clusters, strategies=list(strats.keys()), strategy=strategy, featurizers=config.featurizers, url=url)


@app.route('/annotate')
def annotate():
    """
    For hand-annotating comments for a given article.
    """
    return render_template('annotate.html')


@app.route('/api/annotate', methods=['POST'])
def save_annotations():
    """
    Save the annotations from the frontend.
    """
    data = request.get_json()
    title = data['subject']['title']

    with open('data/annotations/{0}.json'.format(title), 'w') as f:
        json.dump(data, f)

    return jsonify({
        'success': True
    })


# Some simple API routes for an async frontend.
@app.route('/api/comments', methods=['GET'])
def get_comments():
    """
    Gets asset data (title, body) and comments for a given NYT article url.
    """
    url = request.args['url']
    if url:
        asset = services.get_asset(url)['result']

        if asset is None:
            raise Exception('Couldn\'t find an asset matching the url {0}'.format(url))
        elif 'article' in asset:
            body = asset['article']['body']
            title = asset['article']['print_information']['headline']
        elif 'blogpost' in asset:
            body = asset['blogpost']['post_content']
            title = asset['blogpost']['post_title']
        else:
            raise Exception('Unrecognized asset')

        comments = services.get_comments(url, n=300)
        comments = [c for c in comments if len(c.body) > 140]

    else:
        body = '(using example data)'
        title = 'Example Data'

        class FauxComment():
            def __init__(self, body, id):
                self.id = id
                self.body = strip_tags(body)
                self.body_html = body
                self.author = 'some author'
                self.score = 0

        path_to_examples = 'data/examples.json'
        clusters = json.load(open(path_to_examples, 'r'))
        docs = []
        for clus in clusters:
            for doc in clus:
                docs.append(doc)
        comments = [FauxComment(d, i) for i, d in enumerate(docs) if len(doc) > 140]

    return jsonify({
        'body': body,
        'title': title,
        'comments': [{
            'id': c.id,
            'body': c.body,
            'body_html': c.body_html,
            'author': c.author,
            'score': c.score
        } for c in comments]
    })


from geiger.sentences import Sentence
from nltk.tokenize import sent_tokenize
@app.route('/api/geiger', methods=['POST'])
def geigerize():
    """
    Selects highlights from submitted comments
    using the specified strategy.
    """
    data = request.get_json()
    strat = data['strategy']

    # Wrangle posted comments into the minimal format needed for processing.
    comments = [Comment({
        'commentID': c['id'],
        'commentBody': c['body_html'],
        'recommendations': c['score'],
        'userDisplayName': c['author'],
        'createDate': 0,
        'replies': [] # ignoring replies for now
    }) for c in data['comments']]

    resolution = 'sentences'
    results = []

    if resolution == 'sentences':
        # Try out sentences as the object
        sentences = [[Sentence(sent, c) for sent in sent_tokenize(c.body)] for c in comments]
        sentences = [s for sents in sentences for s in sents]

        # Run the specified strategy.
        raw_results = getattr(geiger, strat)(sentences)

        # Format results into something jsonify-able.
        for r in raw_results:
            s = r[1]
            results.append({
                'sentence': r[0],
                'comment': {
                    'id': s.comment.id,
                    'body': s.body,
                    'author': s.comment.author
                },
                'support': int(r[2]),
                'cohort': [c.body for c in r[3]]
            })

    else:
        raw_results = getattr(geiger, strat)(comments)

        # Format results into something jsonify-able.
        for r in raw_results:
            c = r[1]
            results.append({
                'sentence': r[0],
                'comment': {
                    'id': c.id,
                    'body': c.body,
                    'author': c.author
                },
                'support': int(r[2]),
                'cohort': [c.body for c in r[3]]
            })

    return jsonify(results=results)
