import sys
import json
import math
from glob import glob
from collections import defaultdict
from gensim.models import Phrases
from nltk.tokenize import word_tokenize, sent_tokenize
from geiger.services import get_comments
from geiger.util.progress import Progress
from geiger.text.clean import strip_punct
from geiger.text.tokenize import keyword_tokenize


def train_phrases(paths=['data/asset_bodies.txt']):
    """
    Train a bigram phrase model on a list of files.
    """
    n = 0
    for path in paths:
        print('Counting lines for {0}...'.format(path))
        n += sum(1 for line in open(path, 'r'))
    print('Processing {0} lines...'.format(n))

    # Change to use less memory. Default is 40m.
    max_vocab_size = 40000000

    print('Training bigrams...')
    bigram = Phrases(_phrase_doc_stream(paths, n), max_vocab_size=max_vocab_size, threshold=8.)

    print('Saving...')
    bigram.save('data/bigram_model.phrases')

    print('Some examples:')
    docs = [
        ['the', 'new', 'york', 'times', 'is', 'a', 'newspaper'],
        ['concern', 'is', 'rising', 'in', 'many', 'quarters', 'that', 'the', 'united', 'states', 'is', 'retreating', 'from', 'global', 'economic', 'leadership', 'just', 'when', 'it', 'is', 'needed', 'most'],
        ['the', 'afghan', 'president', 'ashraf', 'ghani', 'blamed', 'the', 'islamic', 'state', 'group'],
        ['building', 'maintenance', 'by', 'the', 'hrynenko', 'family', 'which', 'owns', 'properties', 'in', 'the', 'east', 'village'],
        ['a', 'telegram', 'from', 'the', 'american', 'embassy', 'in', 'constantinople', 'to', 'the', 'state', 'department', 'in', 'washington']
    ]
    for r in bigram[docs]:
        print(r)


def train_idf(paths=['data/asset_bodies.txt']):
    """
    Train a IDF model on a list of files.
    """
    n = 0
    for path in paths:
        print('Counting lines for {0}...'.format(path))
        n += sum(1 for line in open(path, 'r'))
    print('Processing {0} lines...'.format(n))

    idf = defaultdict(int)
    N = 0

    for tokens in _idf_doc_stream(paths, n):
        N += 1
        for token in tokens:
            idf[token] += 1


    for k, v in idf.items():
        idf[k] = math.log(N/v)

    with open('data/idf.json', 'w') as f:
        json.dump(idf, f)


def get_comments(url):
    """
    Get all the comments for a given NYT url, save as a JSON dump.
    """
    comments = get_comments(url)

    data = [r for r in get_replies(comments)]
    print(len(data))

    fname = url.replace('/', '_')
    with open('{0}.json'.format(fname), 'w') as f:
        json.dump(data, f)


def _get_replies(comments):
    for c in comments:
        yield {'body': c.body, 'score': c.score}
        if c.replies:
            yield from get_replies(c.replies)


def _phrase_doc_stream(paths, n):
    """
    Generator to feed sentences to the phrase model.
    """
    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)
                for sent in sent_tokenize(line.lower()):
                    tokens = word_tokenize(strip_punct(sent))
                    yield tokens


def _idf_doc_stream(paths, n):
    """
    Generator to feed sentences to IDF model.
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



# Use the following if you split up (parallelize) your IDF computing.
# You have to manually specify N in this case.
def _merge_idfs(N):
    data = glob('data/idf/idf_*.json')
    idfs = [json.load(open(d, 'r')) for d in data]
    return merge(idfs)

def _merge(dicts):
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


if __name__ == '__main__':
    # Convenient, but hacky
    globals()[sys.argv[1]]()
