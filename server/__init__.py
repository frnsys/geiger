from flask import Flask, render_template

import config
from geiger import services
from compare import compare

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/geiger', defaults={'url': None})
@app.route('/geiger/<path:url>')
def geiger(url):
    if url is not None:
        asset = services.get_asset(url)['result']

        if asset is None:
            raise Exception('Couldn\'t find an asset matching the url {0}'.format(url))
        elif 'article' in asset:
            body = asset['article']['body']
        elif 'blogpost' in asset:
            body = asset['blogpost']['post_content']
        else:
            raise Exception('Unrecognized asset')

    else:
        body = '(using example data)'

    results = [(strat, result) for strat, result in compare(url)]

    return render_template('index.html', url=url, subject=body, results=results)

@app.route('/visualize/<path:url>')
def visualize(url):
    comments = services.get_comments(url, n=300)
    comments, tree, stats = examine(comments)

    return render_template('visualize.html', comments=comments, tree=tree, stats=stats, url=url)


import json
from geiger.text import strip_tags
from geiger import clustering
@app.route('/visualize_strat/<strategy>')
def visualize_strat(strategy):
    path_to_examples = 'data/examples.json'
    clusters = json.load(open(path_to_examples, 'r'))
    docs = []
    for clus in clusters:
        for doc in clus:
            docs.append(doc)

    docs = [strip_tags(doc) for doc in docs if len(doc) >= 140] # drop short comments :D

    class FauxComment():
        def __init__(self, body):
            self.body = body

    comments = [FauxComment(d) for d in docs]

    if strategy == 'lda':
        clusters = clustering.lda(comments)
    elif strategy == 'hac':
        clusters = clustering.hac(comments)
    elif strategy == 'ihac':
        clusters = clustering.ihac(comments)
    elif strategy == 'k_means':
        clusters = clustering.k_means(comments)

    return render_template('visualize_strat.html', clusters=clusters)


@app.route('/visualize_strat/<strategy>/<path:url>')
def visualize_strat_url(strategy, url):
    comments = services.get_comments(url, n=300)
    comments = [c for c in comments if len(c.body) > 140]

    if strategy == 'lda':
        clusters = clustering.lda(comments)
    elif strategy == 'hac':
        clusters = clustering.hac(comments)
    elif strategy == 'ihac':
        clusters = clustering.ihac(comments)
    elif strategy == 'k_means':
        clusters = clustering.k_means(comments)

    return render_template('visualize_strat.html', clusters=clusters)
