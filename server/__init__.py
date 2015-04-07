import requests
from requests.auth import HTTPDigestAuth
from flask import Flask, render_template

import config
from comment import Comment
from geiger import highlights

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/geiger', defaults={'url': ''})
@app.route('/geiger/<path:url>')
def geiger(url):
    asset = get_asset(url)['result']

    if asset is None:
        raise Exception('Couldn\'t find an asset matching the url {0}'.format(url))
    elif 'article' in asset:
        body = asset['article']['body']
    elif 'blogpost' in asset:
        body = asset['blogpost']['post_content']
    else:
        raise Exception('Unrecognized asset')

    comments = [Comment(c) for c in get_comments(url, n=300)]

    comments = highlights(comments,
                            min_size=config.min_cluster_size,
                            dist_cutoff=config.distance_cutoff)
    comments.sort(key=lambda c: c.score, reverse=True)

    return render_template('index.html', url=url, subject=body, comments=comments)


def get_asset(asset_url):
    """
    Fetches an asset's data from the Scoop API.
    """
    r = requests.get(config.scoop_base + asset_url, auth=HTTPDigestAuth(*config.scoop_auth))
    return r.json() if r.status_code == 200 else None


def get_comments(asset_url, n=100):
    """
    Fetches up to n comments for an asset from the Community API.
    """
    per_page = 25
    pages = -(-n//per_page) # ceil
    params = {
        'api-key': config.community_key,
        'url': asset_url,
        'offset': 0
    }

    comments = []
    for i in range(pages):
        params['offset'] = i * 25
        r = requests.get(config.community_base, params=params)
        if r.status_code == 200:
            comments += r.json()['results']['comments']
        else:
            break
    return comments
