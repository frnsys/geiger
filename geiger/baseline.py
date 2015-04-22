import re
import json
from itertools import combinations
from nltk.tokenize import sent_tokenize
from geiger.text import keyword_tokenize, gram_size, lemma_forms
from geiger.sentences import prefilter, Sentence


idf = json.load(open('data/idf.json', 'r'))


def highlight(term, doc):
    """
    Highlights each instance of the given term
    in the document. All forms of the term will be highlighted.
    """
    # Determine which forms are present for the term in the document
    if gram_size(term) == 1:
        # Replace longer forms first so we don't replace their substrings.
        forms = sorted(lemma_forms(term, doc), key=lambda f: len(f), reverse=True)
    else:
        forms = [term]

    matches = []
    for t in forms:
        # This captures 'F.D.A' if given 'FDA'
        # yeah, it's kind of overkill
        reg = '[.]?'.join(list(t))

        # Spaces might be spaces, or they might be hyphens
        reg = reg.replace(' ', '[\s-]')

        # Only match the term if it is not continguous with other characters.
        # Otherwise it might be a substring of another word, which we want to
        # ignore
        reg = '(^|{0})({1})($|{0})'.format('[^A-Za-z]', reg)

        matches += [match[1] for match in re.findall(reg, doc, flags=re.IGNORECASE)]

    # Remove dupes
    matches = set(matches)

    for match in matches:
        doc = re.sub(match, '<span class="highlight">{0}</span>'.format(match), doc)

    return doc


def extract_highlights(comments):
    # Filter by minimum comment requirements
    comments = [c for c in comments if len(c.body) > 140]

    # Get sentences, filtered fairly aggressively
    sents = [[Sentence(sent, c) for sent in sent_tokenize(c.body) if prefilter(sent)] for c in comments]
    sents = [sent for s in sents for sent in s]

    print('{0} sentences...'.format(len(sents)))

    senters = []
    for sent in sents:
        tokens = set(keyword_tokenize(sent.body))
        senters.append((sent, tokens))


    # Group sentences by their aspects
    keywords = set()
    keyphrases = set()
    aspect_map = {}
    for combo in combinations(senters, 2):
        s = combo[0]
        s_ = combo[1]
        for t in s[1].intersection(s_[1]):
            if t not in aspect_map:
                aspect_map[t] = set()
            aspect_map[t].add(s[0])
            if gram_size(t) > 1:
                keyphrases.add(t)
            else:
                keywords.add(t)

    # Prune aspects
    # Check if an aspect is subsumed by another in most cases
    for kw in keywords:
        parents = [kw_ for kw_ in keyphrases if kw in kw_]
        for p in parents:
            aspect_map[kw] = aspect_map[kw].difference(aspect_map[p])


    # Take the top 5 supported aspects
    # Prioritize keyphrases over keywords
    highlights = []
    for k in sorted(aspect_map, key=score(aspect_map), reverse=True)[:10]:
        aspect_sents = sorted(aspect_map[k], key=lambda s: s.comment.score, reverse=True)
        aspect_sents = [(sent, highlight(k, sent.body)) for sent in aspect_sents]
        highlights.append((k, aspect_sents[0], aspect_sents[1:]))

    return highlights


def score(aspect_map):
    """
    Emphasize phrases and salient keys (as valued by idf).
    """
    def _score(k):
        # Mean IDF was ~15.2, so slightly bias unencountered terms.
        return idf.get(k, 15.5)**2 * len(aspect_map[k]) * gram_size(k)
    return _score
