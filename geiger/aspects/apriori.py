"""
Apriori Algorithm for text documents.

The original apriori algorithm was mainly for looking at "baskets" (i.e. shopping),
so some terminology may seem weird here. In particular, "transaction" refers to the set of tokens for a document.

See <https://en.wikipedia.org/wiki/Apriori_algorithm>
"""

import string
from itertools import combinations
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def apriori(docs, min_sup=0.3):
    """
    The first pass consists of converting documents
    into "transactions" (sets of their tokens)
    and the initial frequency/support filtering.

    Then iterate until we close in on a final set.

    `docs` can be any iterator or generator so long as it yields lists.
    For example, it can be a list of lists of nouns and noun phrases if trying
    to identify aspects, where each list represents a sentence or document.
    """

    # First pass
    candidates = set()
    transactions = []
    for doc in docs:
        transaction = set(doc)
        candidates = candidates.union({(t,) for t in transaction})
        transactions.append(transaction)
    freq_set = filter_support(candidates, transactions, min_sup)

    # Iterate
    k = 2
    last_set = set()
    while freq_set != set():
        last_set = freq_set
        cands = generate_candidates(freq_set, k)
        freq_set = filter_support(cands, transactions, min_sup)
        k += 1

    return list(flatten(last_set))


def flatten(nested_tuple):
    """
    Flatten nested tuples.
    """
    return tuple([el for tupl in nested_tuple for el in tupl])


def filter_support(candidates, transactions, min_sup):
    """
    Filter candidates to a frequent set by some minimum support.
    """
    counts = defaultdict(lambda: 0)
    for transaction in transactions:
        for c in (c for c in candidates if set(c).issubset(transaction)):
            counts[c] += 1
    return {i for i in candidates if counts[i]/len(transactions) >= min_sup}


def generate_candidates(freq_set, k):
    """
    Generate candidates for an iteration.
    Use this only for k >= 2.
    """
    single_set = {(i,) for i in set(flatten(freq_set))}
    cands = [flatten(f) for f in combinations(single_set, k)]
    return [cand for cand in cands if validate_candidate(cand, freq_set, k)]


def validate_candidate(candidate, freq_set, k):
    """
    Checks if we should keep a candidate.
    We keep a candidate if all its k-1-sized subsets
    are present in the frequent sets.
    """
    for subcand in combinations(candidate, k-1):
        if subcand not in freq_set:
            return False
    return True
