import requests
from requests.auth import HTTPDigestAuth

import config
from geiger.models import Comment


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
            comments += [Comment(c) for c in r.json()['results']['comments']]
        else:
            break
    return comments
