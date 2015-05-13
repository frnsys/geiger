import re
import json
from collections import defaultdict
from itertools import combinations, product
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from geiger.sentences import prefilter, Sentence
from geiger.text.tokenize import keyword_tokenize, gram_size, lemma_forms


stemmer = SnowballStemmer('english')
idf = json.load(open('data/idf.json', 'r'))


def markup_highlights(term, doc):
    """
    Highlights each instance of the given term
    in the document. All forms of the term will be highlighted.
    """
    for term in term.split(','):
        term = term.strip()

        # Determine which forms are present for the term in the document
        if gram_size(term) == 1:
            # Replace longer forms first so we don't replace their substrings.
            forms = sorted(lemma_forms(term, doc), key=lambda f: len(f), reverse=True)
        else:
            forms = [term]

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

            doc = re.sub(reg, '\g<1><span class="highlight">\g<2></span>\g<3>', doc, flags=re.IGNORECASE)

    return doc


def extract_sentences(comments):
    """
    Return qualifying sentences from a list of comments.
    """
    # Get sentences, filtered fairly aggressively
    sents = [[Sentence(sent, c) for sent in sent_tokenize(c.body) if prefilter(sent)] for c in comments]

    # Flatten
    return [sent for s in sents for sent in s]


def extract_highlights(comments):
    sents = extract_sentences(comments)
    print('{0} sentences...'.format(len(sents)))

    # Tokenize sentences,
    # group sentences by their aspects.
    # Keep track of keywords and keyphrases
    keywords = set()
    keyphrases = set()
    aspect_map = defaultdict(set)
    for sent in sents:
        tokens = set(keyword_tokenize(sent.body))
        for t in tokens:
            aspect_map[t].add(sent)
            if gram_size(t) > 1:
                keyphrases.add(t)
            else:
                keywords.add(t)

    # Try to identify novel phrases by looking at
    # keywords which co-occur some percentage of the time.
    # This could probably be more efficient/cleaned up
    for (kw, sents), (kw_, sents_) in combinations(aspect_map.items(), 2):
        intersect = sents.intersection(sents_)

        # Require a min. intersection
        if len(intersect) <= 3:
            continue

        # Look for phrases that are space-delimited or joined by 'and' or '-'
        ph_reg = '({0}|{1})(\s|-)(and\s)?({0}|{1})'.format(kw, kw_)

        # Extract candidate phrases and keep track of their counts
        phrases = defaultdict(int)
        mega_sent = ' '.join(sent.body.lower() for sent in intersect)
        for m in re.findall(ph_reg, mega_sent):
            phrases[''.join(m)] += 1

        if not phrases:
            continue

        # Get the phrase encountered the most
        top_phrase = max(phrases.keys(), key=lambda k: phrases[k])
        top_count = phrases[top_phrase]

        if top_count/len(intersect) >= 0.8:
            # Check if this new phrase is contained by an existing keyphrase.
            if any(top_phrase in ph for ph in keyphrases):
                continue
            aspect_map[top_phrase] = intersect
            keyphrases.add(top_phrase)


    # Prune aspects
    # If a keyword is encountered as part of a keyphrase, remove overlapping
    # sentences with the keyphrase from the keyword's sentences.
    for kw, kp in ((kw, kp) for kw, kp in product(keywords, keyphrases) if kw in kp):
        aspect_map[kw] = aspect_map[kw].difference(aspect_map[kp])

    # Group terms with common stems
    stem_map = defaultdict(list)
    for kw in keywords:
        stem = stemmer.stem(kw)
        stem_map[stem].append(kw)

    # Group sentences with common aspect stems.
    for stem, kws in stem_map.items():
        if len(kws) == 1:
            continue

        key = ', '.join(kws)
        aspect_map[key] = set()
        for kw in kws:
            aspect_map[key] = aspect_map[key].union(aspect_map[kw])

            # Remove the old keys
            aspect_map.pop(kw, None)

    return aspect_map


def select_highlights(aspect_map, top_n=10):
    """
    Rank aspects by support/interestingness
    and return the top n.

    Highlights are returned as:

        [(keyword/keyphrase, representative sentence, cohort sentences), ...]

    """
    highlights = []
    for k in sorted(aspect_map, key=score(aspect_map), reverse=True)[:top_n]:
        aspect_sents = sorted(aspect_map[k], key=lambda s: s.comment.score, reverse=True)
        if not aspect_sents:
            continue
        aspect_sents = [(sent, markup_highlights(k, sent.body)) for sent in aspect_sents]
        highlights.append((k, aspect_sents[0], aspect_sents[1:]))

    return highlights


def score(aspect_map, min_sup=5):
    """
    Emphasize phrases and salient keys (as valued by idf).
    """
    def _score(k):
        support = len(aspect_map[k])

        # Require some minimum support.
        if support < min_sup:
            return 0

        scores = []
        for k_ in k.split(', '):
            # Mean IDF was ~15.2, so slightly bias unencountered terms.
            scores.append(idf.get(k_, 15.5)**2 * support * gram_size(k_))
        return sum(scores)/len(scores)
    return _score
