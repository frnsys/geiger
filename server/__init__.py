import json
from geiger import services
from geiger.comment import Comment
from flask import Flask, render_template, request, jsonify
from flask.json import JSONEncoder
from geiger.aspects import extract_highlights, select_highlights
from geiger.semsim import SemSim
from geiger.baseline import baseline, descriptor, highlight
from geiger.evaluate import evaluate as eval


app = Flask(__name__, static_folder='static', static_url_path='')


class GeigerJSONEncoder(JSONEncoder):
    """
    Custom JSONEncoder which checks if a
    class implements `to_json`, using it if it is found.
    """
    def default(self, obj):
        # First check if a `to_json` method is available,
        # since that is the trump card
        if hasattr(obj, 'to_json'):
            return obj.to_json()
        try:
            # Strings are technically iterable but we want to return them as is.
            # Otherwise they are recursively iterable
            if isinstance(obj, str):
                return obj
            iterable = iter(obj)
            return [self.default(o) for o in obj]
        except TypeError:
            pass
        return self._encode(obj)

    def _encode(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json()
        return super().default(obj)
app.json_encoder = GeigerJSONEncoder


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/evaluate')
def evaluate():
    datasets = [
        'climate_change'
    ]
    return render_template('evaluate.html', datasets=datasets)


@app.route('/talked-about', defaults={'url':''})
@app.route('/talked-about/<path:url>')
def talked_about_preview(url):
    title, body, comments = _fetch_asset(url)
    highlights = extract_highlights(comments)
    highlights = select_highlights(highlights, top_n=20)
    return render_template('talked_about.html', highlights=highlights)


# Some simple API routes for an async frontend
@app.route('/api/comments', methods=['GET'])
def get_comments():
    """
    Gets asset data (title, body) and comments for a given NYT article url.
    """
    url = request.args['url']
    title, body, comments = _fetch_asset(url)

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


@app.route('/api/geiger', methods=['POST'])
def geigerize():
    data = request.get_json()

    # Wrangle posted comments into the minimal format needed for processing
    comments = [Comment({
        'commentID': c['id'],
        'commentBody': c['body_html'],
        'recommendations': c['score'],
        'userDisplayName': c['author'],
        'createDate': 0,
        'replies': [] # ignoring replies for now
    }) for c in data['comments']]

    # Remove duplicates
    bodies = list({c.body for c in comments})

    #from nltk import sent_tokenize
    #from geiger.sentences import prefilter
    #_sents = [sent_tokenize(b) for b in bodies]
    #sents = []
    #for sent_grp in _sents:
        #sents += [s for s in sent_grp if prefilter(s)]

    semsim = SemSim(debug=True)
    clusters, descriptors = semsim.cluster(bodies)
                                           #eps=[3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                                #4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,
                                                #5.0, 5.1, 5.2, 5.3, 5.5, 5.5, 5.6, 5.7, 5.8, 5.9])


    # Get salient terms
    all_terms = sorted(list(semsim.all_terms), key=lambda t: t.salience, reverse=True)

    # Remove duplicates
    descriptors = [list(set(desc)) for desc in descriptors]

    return jsonify(results={
        'clusters': list(zip(clusters, descriptors)),
        'terms': all_terms,
        'gidf': {t.term: t.gidf for t in semsim.all_terms},
        'lidf': {t.term: t.iidf for t in semsim.all_terms},
        'saliences': {t.term: t.salience for t in semsim.all_terms}
    })


@app.route('/api/baseline', methods=['POST'])
def baseline_cluster():
    data = request.get_json()

    # Wrangle posted comments into the minimal format needed for processing
    comments = [Comment({
        'commentID': c['id'],
        'commentBody': c['body_html'],
        'recommendations': c['score'],
        'userDisplayName': c['author'],
        'createDate': 0,
        'replies': [] # ignoring replies for now
    }) for c in data['comments']]

    # Remove duplicates
    bodies = list({c.body for c in comments})

    clusters = baseline(bodies)
                        #eps=[1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                             #2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                             #3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9])

    descriptors = [descriptor(clus) for clus in clusters]
    highlighted = [highlight(clus) for clus in clusters]

    return jsonify(results={
        'clusters': clusters,
        'descriptors': descriptors,
        'highlighted_clusters': highlighted
    })


@app.route('/api/talked-about', methods=['POST'])
def talked_about():
    data = request.get_json()

    # Wrangle posted comments into the minimal format needed for processing
    comments = [Comment({
        'commentID': c['id'],
        'commentBody': c['body_html'],
        'recommendations': c['score'],
        'userDisplayName': c['author'],
        'createDate': 0,
        'replies': [] # ignoring replies for now
    }) for c in data['comments']]
    highlights = extract_highlights(comments)
    highlights = select_highlights(highlights)

    # Format results into something jsonify-able
    results = []
    for r in highlights:
        sent, body = r[1]
        results.append({
            'aspect': r[0],
            'sentence': body,
            'comment': {
                'id': sent.comment.id,
                'body': sent.comment.body,
                'author': sent.comment.author
            },
            'support': len(r[2]),
            'cohort': [b for s, b in r[2]]
        })

    return jsonify(results=results)


@app.route('/api/evaluate', methods=['POST'])
def evaluate_api():
    data = request.get_json()
    dataset = 'data/evaluate/{}.txt'.format(data['dataset'])
    kw_results, clus_results, docs, all_terms, pruned, true_labels, pred_labels, summary = eval(dataset)

    # Get salient terms
    all_terms = sorted(list(all_terms), key=lambda t: t.salience, reverse=True)

    return jsonify(results={
        'docs': docs,
        'terms': all_terms,
        'pruned': pruned,
        'keywords': kw_results,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'summary': summary
    })


def _fetch_asset(url):
    """
    Fetch an asset and its comments,
    using the example data if the url is empty.
    """
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

        path_to_examples = 'data/examples/climate_example.json'
        #path_to_examples = 'data/examples/clinton_example.json'
        #path_to_examples = 'data/examples/gaymarriage_example.json'
        data = json.load(open(path_to_examples, 'r'))
        comments = [Comment({
            'commentID': i,
            'commentBody': d['body'],
            'recommendations': d['score'],
            'userDisplayName': '[the author]',
            'createDate': '1431494183',
            'replies': []
        }) for i, d in enumerate(data) if len(d['body']) > 140]

    return title, body, comments
