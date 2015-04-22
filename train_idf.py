import math
import json
from glob import glob
from collections import defaultdict
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

idf = defaultdict(int)
N = 0

for tokens in doc_stream(paths, n):
    N += 1
    for token in tokens:
        idf[token] += 1


for k, v in idf.items():
    idf[k] = math.log(N/v)

with open('data/idf.json', 'w') as f:
    json.dump(idf, f)


# Use the following if you split up your IDF computing.
# You have to manually specify N in this case.

def merge_idfs(N):
    data = glob('data/idf/idf_*.json')
    idfs = [json.load(open(d, 'r')) for d in data]
    return merge(idfs)

def merge(dicts):
    i = 0
    n = sum([len(d.keys()) for d in dicts])
    p = Progress()
    merged = defaultdict(int)
    for d in dicts:
        for k in d:
            i += 1
            p.print_progress(i/n)
            merged[k] += d[k]
    return merged
