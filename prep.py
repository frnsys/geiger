import os
import sys
import json
import math
from glob import glob
from itertools import chain, islice
from collections import Counter, defaultdict
from gensim.models import Phrases
from nltk.tokenize import word_tokenize, sent_tokenize
from geiger.services import get_comments
from geiger.util.progress import Progress
from geiger.text.clean import strip_punct
from geiger.text.tokenize import keyword_tokenize
from geiger.util.parallel import parallelize


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
    Train a IDF model on a list of files (parallelized).
    """
    for path in paths:
        args = [(file,) for file in _split_file(path, chunk_size=5000)]

    results = parallelize(_count_idf, args)
    idfs, n_docs = zip(*results)

    print('Merging...')
    idf = _merge(idfs)
    N = sum(n_docs)

    for k, v in idf.items():
        idf[k] = math.log(N/v)

    with open('data/idf.json', 'w') as f:
        json.dump(idf, f)


def _count_idf(path):
    """
    Count term frequencies and documents for a single file.
    """
    N = 0
    idf = defaultdict(int)
    for tokens in _idf_doc_stream(path):
        N += 1
        # Don't count freq, just presence
        for token in set(tokens):
            idf[token] += 1
    return idf, N


def train_tf(paths=['data/asset_bodies.txt']):
    """
    Train a map of term frequencies on a list of files (parallelized).
    """
    for path in paths:
        args = [(file,) for file in _split_file(path, chunk_size=5000)]

    results = parallelize(_count_tf, args)

    print('Merging...')
    tf = _merge(results)

    with open('data/tf.json', 'w') as f:
        json.dump(tf, f)


def _count_tf(path):
    """
    Count term frequencies for a single file.
    """
    tf = defaultdict(int)
    for tokens in _tf_doc_stream(path):
        for token in tokens:
            tf[token] += 1
    return tf


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


def _idf_doc_stream(path):
    """
    Generator to feed sentences to IDF trainer.
    """
    with open(path, 'r') as f:
        for line in f:
            yield keyword_tokenize(line)
            #for sent in sent_tokenize(line):
                #yield keyword_tokenize(sent)


def _tf_doc_stream(path):
    """
    Generator to feed sentences to TF trainer.
    """
    with open(path, 'r') as f:
        for line in f:
            #yield keyword_tokenize(line)
            yield word_tokenize(line.lower())


def _merge(dicts):
    """
    Merges a list of dicts, summing their values.
    """
    merged = sum([Counter(d) for d in dicts], Counter())
    return dict(merged)


def _chunks(iterable, n):
    """
    Splits an iterable into chunks of size n.
    """
    iterable = iter(iterable)
    while True:
        # store one line in memory,
        # chain it to an iterator on the rest of the chunk
        yield chain([next(iterable)], islice(iterable, n-1))


def _split_file(path, chunk_size=50000):
    """
    Splits the specified file into smaller files.
    """
    with open(path) as f:
        for i, lines in enumerate(_chunks(f, chunk_size)):
            file_split = '{}.{}'.format(os.path.basename(path), i)
            chunk_path = os.path.join('/tmp', file_split)
            with open(chunk_path, 'w') as f:
                f.writelines(lines)
            yield chunk_path


if __name__ == '__main__':
    # Convenient, but hacky
    globals()[sys.argv[1]]()
