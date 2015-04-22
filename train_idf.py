import math
import json
from nltk.tokenize import sent_tokenize
from geiger.util.progress import Progress
from geiger.text import keyword_tokenize

paths = ['data/asset_bodies.txt']

n = 0
for path in paths:
    print('Counting lines for {0}...'.format(path))
    n += sum(1 for line in open(path, 'r'))
print('Processing {0} lines...'.format(n))

def doc_stream(paths, n):
    """
    Generator to feed sentences to Doc2Vec.
    """

    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)
                for sent in sent_tokenize(line):
                    yield keyword_tokenize(sent)

idf = {}
N = 0

for tokens in doc_stream(paths, n):
    N += 1
    for token in tokens:
        idf[token] = idf.get(token, 0) + 1

for k, v in idf.items():
    idf[k] = math.log(N/v)

with open('idf.json', 'w') as f:
    json.dump(idf, f)
