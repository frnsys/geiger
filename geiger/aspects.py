import random
from collections import namedtuple
from nltk import pos_tag, word_tokenize, sent_tokenize, RegexpParser
from nltk.corpus import stopwords
from geiger.text import strip_tags
from geiger.keywords import Rake

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
        sents += [sent for sent in extract_aspects(c)]

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
    for aspect, sents in aspects.items():
        print('[{0}] {1}'.format(aspect, random.choice(sents).text))


def extract_aspects(comment, strategy='pos_tag'):
    """
    Extracts all aspects for each sentence in comment.

    Args:
        | comment       -- Comment
        | strategy      -- str, specify how aspects will be extracted.
                            Options are ['pos_tag', 'rake'].
    """
    for sent in sent_tokenize(strip_tags(comment.body)):
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
