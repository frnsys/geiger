import json
from broca import Pipeline
from geiger import services
from geiger.models import Comment
from broca.vectorize import BoW
from broca.tokenize.keyword import Overkill
from broca.preprocess import Cleaner, HTMLCleaner
from geiger.pipes import LDA, SemSim, HSCluster, DBSCAN, AspectCluster, Distance
from nltk.tokenize import sent_tokenize
from geiger.sentences import prefilter
from geiger.pipes.aspect import markup_highlights
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/')
def index():
    return render_template('index.html')


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
            'author': c.author,
            'score': c.score
        } for c in comments]
    })


@app.route('/api/cluster', methods=['POST'])
def cluster():
    data = request.get_json()

    # Wrangle posted comments into the minimal format needed for processing
    comments = [Comment({
        'commentID': c['id'],
        'commentBody': c['body'],
        'recommendations': c['score'],
        'userDisplayName': c['author'],
        'createDate': 0,
        'replies': [] # ignoring replies for now
    }) for c in data['comments']]

    # Remove duplicates
    docs = list({c.body for c in comments})

    preprocess = Pipeline(
        HTMLCleaner(),
        Cleaner()
    )

    names = [
        'lda_hscluster',
        'lda_dbscan',
        'semsim_hscluster',
        'semsim_dbscan',
        'bow_hscluster',
        'bow_dbscan',
        'aspects'
    ]
    pipelines = [
        Pipeline(
            preprocess,
            BoW(),
            LDA(n_topics=10),
            Distance(metric='euclidean'),
            [HSCluster(), DBSCAN()]
        ),
        Pipeline(
            preprocess,
            Overkill(),
            SemSim(),
            [HSCluster(), DBSCAN()]
        ),
        Pipeline(
            preprocess,
            BoW(),
            Distance(metric='euclidean'),
            [HSCluster(), DBSCAN()]
        ),
    ]

    results = []
    for p in pipelines:
        print('Running pipeline:', p)
        outputs = p(docs)

        doc_clusters = []
        for out in outputs:
            for clus in out:
                clus_docs = []
                for id in clus:
                    clus_docs.append(docs[id])
                doc_clusters.append(clus_docs)
            results.append(doc_clusters)

    # Get sentences, filtered fairly aggressively
    sents = [[sent for sent in sent_tokenize(d) if prefilter(sent)] for d in docs]
    sents = [sent for s in sents for sent in s]
    aspect = Pipeline(
        preprocess,
        Overkill(),
        AspectCluster()
    )

    output = aspect(sents)
    highlighted = []
    for k, aspect_sents in output:
        highlighted.append([markup_highlights(k, sents[i]) for i in aspect_sents])
    results.append(highlighted)

    return jsonify(results=dict(zip(names, results)))


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

        path_to_examples = 'examples/climate_example.json'
        #path_to_examples = 'examples/clinton_example.json'
        #path_to_examples = 'examples/gaymarriage_example.json'
        data = json.load(open(path_to_examples, 'r'))
        comments = [Comment({
            'commentID': i,
            'commentBody': d['body'],
            'recommendations': d['score'],
            'userDisplayName': '[the author]',
            'createDate': '1431494183',
            'replies': []
        }) for i, d in enumerate(data) if len(d['body']) > 140][:100]

    return title, body, comments
