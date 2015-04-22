import re
from itertools import combinations
from nltk.tokenize import sent_tokenize
from geiger.text import keyword_tokenize, gram_size, lemma_forms
from geiger.sentences import prefilter


def highlight(term, doc):
    """
    Highlights each instance of the given term
    in the document. All forms of the term will be highlighted.
    """

    # Determine which forms are present for the term in the document
    if gram_size(term) == 1:
        forms = lemma_forms(term, doc)
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
    docs = [c.body for c in comments]
    docs = [doc for doc in docs if len(doc) > 140]

    # Get sentences, filtered fairly aggressively
    sents = [[sent for sent in sent_tokenize(doc) if prefilter(sent)] for doc in docs]
    sents = [sent for s in sents for sent in s]

    print('{0} sentences...'.format(len(sents)))

    senters = []
    for sent in sents:
        tokens = set(keyword_tokenize(sent))
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
    for k in sorted(aspect_map, key=lambda k: gram_size(k) * len(aspect_map[k]), reverse=True)[:10]:
        highlights.append((k, [highlight(k, sent) for sent in aspect_map[k]]))

    return highlights
