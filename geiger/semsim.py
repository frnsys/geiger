import math
import operator
import numpy as np
from functools import reduce
from itertools import product, combinations
from nltk.tokenize import sent_tokenize
from geiger.text import keyword_tokenize
from geiger.sentences import prefilter, Sentence
from geiger.util.progress import Progress
from gensim.models.word2vec import Word2Vec

#print('Loading word2vec model...')
#w2v = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)


def semsim(comments, sim_cutoff=0.8):
    """
    Cluster sentences by "semsim" (TM).
    """
    # Filter by minimum comment requirements
    comments = [c for c in comments if len(c.body) > 140]

    # Get sentences, filtered fairly aggressively
    sents = [[Sentence(sent, c) for sent in sent_tokenize(c.body) if prefilter(sent)] for c in comments]
    sents = [sent for s in sents for sent in s]

    print('{0} sentences...'.format(len(sents)))

    print('Calculating similarity matrix...')
    sim_mat = build_sim_mat([s.body for s in sents])

    print('Running HAC...')
    labels = hac(sents, sim_mat, sim_cutoff)

    clus = {}
    for i, label in enumerate(labels):
        if label not in clus:
            clus[label] = []
        clus[label].append(sents[i])

    # Sort clusters by size
    clusters = clus.values()
    clusters = sorted(clusters, key=lambda c: len(c), reverse=True)[:5]

    results = []
    for clus in clusters:
        clus = sorted(clus, key=lambda s: s.comment.score, reverse=True)
        sent = clus[0]
        results.append((sent.body, sent.comment, len(clus), clus[1:]))

    return results


def sim(d_1, d_2, idfs, conn):
    """
    This is designed to be symmetric, i.e. `sim(d_1, d_2) == sim(d_2, d_1)`.
    """
    n = len(d_1)
    m = len(d_2)

    # TO DO this can be more efficient since the most sim is a symmetric
    # relation.
    max_sims = []
    for t in d_1:
        sims = []
        for t_ in d_2:
            try:
                #sim = w2v.similarity(t, t_)
                conn.send((t, t_))
                sim = conn.recv()

            # If the doc2vec model encounters an unrecognized word,
            # it raises a KeyError.
            except KeyError:
                sim = 0.
            sims.append(sim)
        max_sims.append(max(sims) * tfidf(t, d_1, idfs))

    for t in d_2:
        sims = []
        for t_ in d_1:
            try:
                #sim = w2v.similarity(t, t_)
                conn.send((t, t_))
                sim = conn.recv()

            # If the doc2vec model encounters an unrecognized word,
            # it raises a KeyError.
            except KeyError:
                sim = 0.
            sims.append(sim)
        max_sims.append(max(sims) * tfidf(t, d_2, idfs))

    return sum(max_sims)/(m+n)


def tf(term, doc):
    """
    Calculate term-frequency.
    Here, `doc` is a list of tokens.
    """
    count = sum([1 for t in doc if term == t])

    counts = {}
    for t in doc:
        counts[t] = counts.get(t, 0) + 1

    return count/max(counts.values())


def idf(term, docs):
    """
    Calculate inverse document-frequency.
    Here, `docs` is a list of documents,
    each represented as a list of tokens.
    """
    N = len(docs)
    n = sum([1 for d in docs if term in d])
    return math.log(N/n)


def tfidf(term, doc, idf_map):
    """
    Calculate TF-IDF.
    Requires a precomputed term->idf mapping (as a dict).
    """
    return tf(term, doc) * idf_map[term]


def build_idf_map(docs):
    """
    Builds a term->idf mapping.
    Here, `docs` is a list of documents,
    each represented as a string.
    """
    tokens = [t for doc in docs for t in doc]
    return {t: idf(t, docs) for t in tokens}


def hac(docs, sim_mat, threshold):
    """
    Hierarchical Agglomerative Clustering.

    Args:
        - docs          -- list of documents as strings.
        - sim_mat       -- precomputed pairwise similarity matrix b/w docs.
        - threshold     -- minimum similarity threshold before clustering
                            terminates.

    This does not actually maintain a hierarchy, so perhaps better to just
    call it "agglomerative clustering"...
    """

    def _sim(a, b):
        """
        The similarity b/w two clusters is
        the the minimimum similarity of members between
        the two clusters.

        This linkage method is known as "minimum",
        it's stricter about cluster membership.
        See <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>
        for a good overview on other linkage methods.
        """
        sims = [sim_mat[pair] for pair in product(a, b)]
        return min(sims)

    size = len(docs)
    labels = [i for i in range(size)]

    # Each item starts in its own cluster.
    clusters = {(i,) for i in range(size)}

    # Initialize the label.
    j = len(clusters) + 1
    while True:
        pairs = combinations(clusters, 2)

        # Calculate the similarity for each pair.
        # (pair, similarity)
        scores = [(p, _sim(*p)) for p in pairs]

        # Get the highest similarity to determine which pair is merged.
        mxm = max(scores, key=operator.itemgetter(1))

        # Stop if the highest similarity is below the threshold.
        if mxm[1] < threshold:
            break

        # Remove the to-be-merged pair from the set of clusters,
        # then merge (flatten) them.
        pair = mxm[0]
        clusters -= set(pair)
        flat_pair = reduce(lambda x,y: x + y, pair)

        # Update the labels for the pairs' members.
        for i in flat_pair:
            labels[i] = j

        # Add the new cluster to the clusters.
        clusters.add(flat_pair)

        # If one cluster is left, we can't continue merging.
        if len(clusters) == 1:
            break

        # Increment the label.
        j += 1

    return labels



from multiprocessing.connection import Client
def build_sim_mat(docs):
    address = ('localhost', 6000)
    with Client(address, authkey=b'password') as conn:
        """
        Construct the pairwise similarity matrix for a list of documents (as strings).
        """
        # Tokenize documents.
        docs = [keyword_tokenize(d) for d in docs]

        # Calculate IDF values for later usage.
        idfs = build_idf_map(docs)

        # Initialize with zeros.
        n = len(docs)
        sim_mat = np.zeros((n,n))

        p = Progress('SIMMAT')
        t = n*n
        inc = 0
        for i, doc in enumerate(docs):
            for j, doc_ in enumerate(docs):
                inc += 1
                p.print_progress(inc/t)
                if i == j:
                    continue
                elif sim_mat[j,i] != 0.:
                    sim_mat[i,j] = sim_mat[j,i]
                    continue
                else:
                    sim_mat[i,j] = sim(doc, doc_, idfs, conn)

    return sim_mat


