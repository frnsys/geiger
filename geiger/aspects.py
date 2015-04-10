import random
from collections import namedtuple
from nltk import pos_tag, word_tokenize, sent_tokenize, RegexpParser
from nltk.corpus import stopwords
from geiger.keywords import Rake

"""
The code for the NLTK noun phrase aspect extraction strategy is adapted from Jeffrey Fossett:
<https://github.com/Fossj117/opinion-mining/blob/master/classes/transformers/asp_extractors.py>
"""

# Grammar for NP chunking
GRAMMAR = r"""
NBAR:
    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

NP:
    {<NBAR><IN|CC><NBAR>}  # Above, connected with in/of/etc...
    {<NBAR>}
"""
CHUNKER = RegexpParser(GRAMMAR)
r = Rake('data/SmartStoplist.txt')
stops = stopwords.words('english')


Sentence = namedtuple('Sentence', ['text', 'aspects', 'parent'])


def summarize_aspects(comments, strategy='pos_tag'):
    sents = []
    for c in comments:
        sents += [sent for sent in extract_aspects(c, strategy)]

    # Calculate support for each aspect.
    counts = {}
    for sent in sents:
        for aspect in sent.aspects:
            if aspect not in counts:
                counts[aspect] = 0
            counts[aspect] += 1

    # Sort and get top n aspects.
    count_sorted = sorted(counts.items(), key=lambda c: c[1], reverse=True)
    top_aspects = [k[0] for k in count_sorted[:5]]

    # Find sentences for each aspect.
    aspects = {k: [] for k in top_aspects}
    for sent in sents:
        overlap = set(sent.aspects).intersection(top_aspects)
        for aspect in overlap:
            aspects[aspect].append(sent)

    # Pick a random sentence for each aspect.
    results = []
    for aspect, sents in aspects.items():
        # NOTE here the support value is not the # of comments, but the # of sentences.
        results.append((random.choice(sents).text, counts[aspect]))
    return results


def extract_aspects(comment, strategy='pos_tag'):
    """
    Extracts all aspects for each sentence in comment.

    Args:
        | comment       -- Comment
        | strategy      -- str, specify how aspects will be extracted.
                            Options are ['pos_tag', 'rake'].
    """
    for sent in sent_tokenize(comment.body):
        if strategy == 'rake':
            # Should we keep the keyword scores?
            # May need to strip puncuation here...getting "don" instead of "don't"
            keywords = r.run(sent)
            yield Sentence(sent, [kw[0] for kw in keywords if kw not in stops], comment)

        else:
            tokens = word_tokenize(sent)
            tagged = pos_tag(tokens)
            tree = CHUNKER.parse(tagged)
            aspects = [[w for w,t in leaf] for leaf in leaves(tree)]
            aspects = [i for s in aspects for i in s if i not in stops]
            yield Sentence(sent, aspects, comment)


def leaves(tree):
    for subtree in tree.subtrees(filter=lambda t: t.label()=='NP'):
        yield subtree.leaves()