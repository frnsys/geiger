from flask import Flask, render_template

import config
from geiger import highlights, services, examine

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/geiger', defaults={'url': ''})
@app.route('/geiger/<path:url>')
def geiger(url):
    asset = services.get_asset(url)['result']

    if asset is None:
        raise Exception('Couldn\'t find an asset matching the url {0}'.format(url))
    elif 'article' in asset:
        body = asset['article']['body']
    elif 'blogpost' in asset:
        body = asset['blogpost']['post_content']
    else:
        raise Exception('Unrecognized asset')

    comments = services.get_comments(url, n=300)

    comments, stats = highlights(comments,
                            min_size=config.min_cluster_size,
                            dist_cutoff=config.distance_cutoff)
    comments.sort(key=lambda c: c[0].score, reverse=True)

    return render_template('index.html', url=url, subject=body, comments=comments, stats=stats)

@app.route('/visualize/<path:url>')
def visualize(url):
    comments = services.get_comments(url, n=300)
    comments, tree, stats = examine(comments)

    return render_template('visualize.html', comments=comments, tree=tree, stats=stats, url=url)

@app.route('/visualize_lda')
def visualize_lda():
    from run_lda import main
    clusters = main()

    return render_template('visualize_lda.html', clusters=clusters)
