import sys
import json
from geiger.text import strip_tags
from geiger.featurizers import featurize
from geiger import clustering, sentences, services

if len(sys.argv) > 1:
    print('Fetching comments...')
    url = sys.argv[1]
    comments = services.get_comments(url, n=300)
    comments = [c for c in comments if len(c.body) > 140]

else:
    print('Loading example comments...')
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

print('Using {0} comments.'.format(len(comments)))

# LDA + extract_by_topics
print('Training topic model...')
clusters, lda = clustering.lda(comments, n_topics=None)
results = sentences.extract_by_topics(clusters, lda)
print(results)

#featurize(comments) # build features for later use
#clusters, lda = clustering.lda(comments, n_topics=13)
#results = sentences.extract_by_distance(clusters)
