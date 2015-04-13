from nltk import pos_tag, word_tokenize, sent_tokenize, RegexpParser
from nltk.corpus import stopwords
from geiger.keywords import Rake

"""
The code for the NLTK noun phrase aspect extraction strategy is adapted from Jeffrey Fossett:
<https://github.com/Fossj117/opinion-mining/blob/master/classes/transformers/asp_extractors.py>
"""
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


def extract_aspects(sentence, strategy='pos_tag'):
    """
    Extracts all aspects for a sentence.

    Args:
        | sentence      -- str
        | strategy      -- str, specify how aspects will be extracted.
                            Options are ['pos_tag', 'rake'].
    """
    if strategy == 'rake':
        # Should we keep the keyword scores?
        # May need to strip puncuation here...getting "don" instead of "don't"
        keywords = r.run(sentence)
        return [kw[0] for kw in keywords if kw not in stops]

    else:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        tree = CHUNKER.parse(tagged)
        aspects = [[w for w,t in leaf] for leaf in _leaves(tree)]
        aspects = [i for s in aspects for i in s if i not in stops]
        return aspects


def _leaves(tree):
    for subtree in tree.subtrees(filter=lambda t: t.label()=='NP'):
        yield subtree.leaves()
