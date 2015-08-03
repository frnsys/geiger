import re
import config
from broca import Pipe
from broca.common.util import gram_size, penn_to_wordnet
from collections import defaultdict
from textblob import Blobber
from itertools import product
from textblob_aptagger import PerceptronTagger
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from geiger.knowledge import IDF

lem = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
blob = Blobber(pos_tagger=PerceptronTagger())
idf = IDF(remote=config.remote)


class AspectCluster():
    input = Pipe.type.tokens
    output = Pipe.type.clusters

    def __call__(self, token_docs):
        aspect_map = self.extract_highlights(token_docs)
        return self.select_highlights(aspect_map)

    def extract_highlights(self, token_docs):
        print('{0} docs...'.format(len(token_docs)))

        # Tokenize sentences,
        # group sentences by their aspects.
        # Keep track of keywords and keyphrases
        keywords = set()
        keyphrases = set()
        aspect_map = defaultdict(set)
        for id, tokens in enumerate(token_docs):
            tokens = set(tokens)
            for t in tokens:
                aspect_map[t].add(id)
                if gram_size(t) > 1:
                    keyphrases.add(t)
                else:
                    keywords.add(t)

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


    def select_highlights(self, aspect_map):
        """
        Rank aspects by support/interestingness
        and return the top n.

        Highlights are returned as:

            [(keyword/keyphrase, representative sentence, cohort sentences), ...]

        """
        highlights = []
        for k in sorted(aspect_map, key=self.score(aspect_map), reverse=True):
            #aspect_sents = sorted(aspect_map[k], key=lambda s: s.comment.score, reverse=True)
            aspect_sents = aspect_map[k]
            if not aspect_sents:
                continue
            #aspect_sents = [(sent, markup_highlights(k, sent)) for sent in aspect_sents]
            highlights.append((k, aspect_sents))

        return highlights


    def score(self, aspect_map, min_sup=5):
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
            reg_ = '[.]?'.join(list(t))

            # Spaces might be spaces, or they might be hyphens
            reg_ = reg_.replace(' ', '[\s-]')

            # Only match the term if it is not continguous with other characters.
            # Otherwise it might be a substring of another word, which we want to
            # ignore
            reg = '(^|{0})({1})($|{0})'.format('[^A-Za-z]', reg_)

            if re.findall(reg, doc):
                doc = re.sub(reg, '\g<1><span class="highlight">\g<2></span>\g<3>', doc, flags=re.IGNORECASE)
            else:
                # If none of the term was found, try with extra alpha characters
                # This helps if a phrase was newly learned and only assembled in
                # its lemma form, so we may be missing the actual form it appears in.
                reg = '(^|{0})({1}[A-Za-z]?)()'.format('[^A-Za-z]', reg_)
                doc = re.sub(reg, '\g<1><span class="highlight">\g<2></span>\g<3>', doc, flags=re.IGNORECASE)

    return doc


def lemma_forms(lemma, doc):
    """
    Extracts all forms for a given term in a given document.
    """
    blo = blob(doc)

    results = []
    for tok, tag in blo.tags:
        wn_tag = penn_to_wordnet(tag)
        if wn_tag is None:
            continue
        l = lem.lemmatize(tok, wn_tag)
        if l != lemma:
            continue
        results.append(tok)
    return results
