import re
from collections import defaultdict
from textblob import Blobber
from textblob_aptagger import PerceptronTagger
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nytnlp.keywords import rake
from nytnlp.util import penn_to_wordnet
from geiger.text.clean import clean_doc
from geiger.knowledge import Bigram


blob = Blobber(pos_tagger=PerceptronTagger())
stops = stopwords.words('english')
lem = WordNetLemmatizer()

import config
bigram = Bigram(remote=config.remote)


def keyword_tokenize(doc):
    """
    Tokenizes a document so that only keywords and phrases
    are returned. Keywords are returned as lemmas.
    """
    doc = clean_doc(doc)
    blo = blob(doc)

    # Only process tokens which are keywords
    kws = rake.extract_keywords(doc)

    # Split phrase keywords into 1gram keywords,
    # to check tokens against
    kws_1g = [kw.split(' ') for kw in kws]
    kws_1g = [kw for grp in kws_1g for kw in grp]

    # Extract keyphrases
    phrases = [ph.replace('_', ' ') for ph in bigram[blo.words]]
    phrases = [ph for ph in phrases if gram_size(ph) > 1]
    phrases += [ph for ph in blo.noun_phrases if gram_size(ph) > 1 and ph not in phrases]

    toks = []
    for tok, tag in blo.tags:
        if tok not in stops and tok in kws_1g:
            wn_tag = penn_to_wordnet(tag)
            if wn_tag is not None:
                toks.append(lem.lemmatize(tok, wn_tag))
    toks += phrases

    return toks


def lemma_forms(lemma, doc):
    """
    Extracts all forms for a given term in a given document.
    """
    doc = clean_doc(doc)
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


def gram_size(term):
    """
    Convenience func for getting n-gram length.
    """
    return len(term.split(' '))


def extract_phrases(docs, raw_docs):
    """
    Learn novel phrases by looking at co-occurrence of candidate term pairings.

    Docs should be input in tokenized (`docs`) and untokenized (`raw_docs`) form.
    """
    # Gather existing keyphrases
    keyphrases = set()
    for doc in docs:
        for t in doc:
            if gram_size(t) > 1:
                keyphrases.add(t)

    # Count document co-occurrences
    t_counts = defaultdict(int)
    pair_docs = defaultdict(list)
    for i, terms in enumerate(docs):
        # We dont convert the doc to a set b/c we want to preserve order
        # Iterate over terms as pairs
        for pair in zip(terms, terms[1:]):
            t_counts[pair] += 1
            pair_docs[pair].append(i)

    # There are a lot of co-occurrences, filter down to those which could
    # potentially be phrases.
    t_counts = {kw: count for kw, count in t_counts.items() if count >= 3}

    # Identify novel phrases by looking at
    # keywords which co-occur some percentage of the time.
    # This could probably be more efficient/cleaned up
    for (kw, kw_), count in t_counts.items():
        # Look for phrases that are space-delimited or joined by 'and' or '-'
        ph_reg = re.compile('({0}|{1})(\s|-)(and\s)?({0}|{1})'.format(kw, kw_))

        # Extract candidate phrases and keep track of their counts
        phrases = defaultdict(int)
        phrase_docs = defaultdict(set)
        for i in pair_docs[(kw, kw_)]:
            for m in ph_reg.findall(raw_docs[i].lower()):
                phrases[''.join(m)] += 1
                phrase_docs[''.join(m)].add(i)

        if not phrases:
            continue

        # Get the phrase encountered the most
        top_phrase = max(phrases.keys(), key=lambda k: phrases[k])
        top_count = phrases[top_phrase]

        if top_count/count >= 0.8:
            # Check if this new phrase is contained by an existing keyphrase.
            if any(top_phrase in ph for ph in keyphrases):
                continue
            keyphrases.add(top_phrase)

            # Add the new phrase to each doc it's found in
            for i in phrase_docs[top_phrase]:
                docs[i].append(top_phrase)

    return docs, keyphrases


