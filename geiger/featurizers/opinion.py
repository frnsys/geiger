import re
import numpy as np
from nltk import word_tokenize

# The PerceptronTagger is _way_ faster than NLTK's default tagger,
# and more accurate to boot.
# See <http://stevenloria.com/tutorial-state-of-the-art-part-of-speech-tagging-in-textblob/>.
from textblob import Blobber
from textblob_aptagger import PerceptronTagger
from geiger.util.progress import Progress

"""
to do: put this in requirements.txt
pip install -U textblob
pip install -U git+https://github.com/sloria/textblob-aptagger.git@dev
"""


class Featurizer():
    """
    Builds opinionated-ness features.

    This featurizer is taken from Jeff Fossett's `LiuFeaturizer`, found at
    <https://github.com/Fossj117/opinion-mining/blob/master/classes/transformers/featurizers.py>
    """
    pos_path = 'data/opinion-lexicon-English/positive-words.txt'
    neg_path = 'data/opinion-lexicon-English/negative-words.txt'

    # Regex credit: Chris Potts
    # Regex to match negation tokens
    NEGATION_RE = re.compile("""(?x)(?:
    ^(?:never|no|nothing|nowhere|noone|none|not|
        havent|hasnt|hadnt|cant|couldnt|shouldnt|
        wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
     )$
    )
    |
    n't""")

    # Regex to match punctuation tokens
    PUNCT_RE = re.compile("^[.:;!?]$")

    def __init__(self):
        self.pos_lex = self.load_lexicon(self.pos_path)
        self.neg_lex = self.load_lexicon(self.neg_path)
        self.blobber = Blobber(pos_tagger=PerceptronTagger())

    def load_lexicon(self, path):
        """
        Loads and processes the lexicons.
        """
        lex = set()
        with open(path, 'r', encoding='ISO-8859-2') as f:
            for line in f.readlines():
                if line[0] != ';' and line.strip() != '':
                    lex.add(line.strip())
        return lex

    def featurize(self, comments, return_ctx=False):
        p = Progress('OPIN')
        n = len(comments)

        pos_feats = []
        lex_feats = []
        for i, c in enumerate(comments):
            p.print_progress((i+1)/n)
            comment = c.body
            pos_feats.append(self.pos_featurize(comment))
            lex_feats.append(self.lex_featurize(comment))

        feats = np.hstack([pos_feats, lex_feats])
        if return_ctx:
            return feats, feats
        else:
            return feats

    def pos_featurize(self, comment):
        """
        Builds part-of-speech features.
        """
        tagged = self.blobber(comment).tags
        tags = [t for _, t in tagged]

        nouns = {'NN', 'NNS', 'NNP', 'NNPS'}
        adjs = {'JJ', 'JJR', 'JJS'}
        advbs = {'RB', 'RBR', 'RBS'}
        pronouns = {'PRP', 'PRP$'}

        n_total = len(tags) if len(tags) != 0 else 1
        n_nouns = len([t for t in tags if t in nouns])
        n_adjs = len([t for t in tags if t in adjs])
        n_advbs = len([t for t in tags if t in advbs])

        # Indicators for special tags
        has_pronoun = 1 if any([t in pronouns for t in tags]) else 0
        has_cardinal = 1 if any ([t == 'CD' for t in tags]) else 0
        has_modal = 1 if any([t == 'MD' and w != 'will' for w, t in tagged]) else 0

        return np.array([
            n_nouns,
            n_adjs,
            n_advbs,
            n_nouns/n_total,
            n_adjs/n_total,
            n_advbs/n_total,
            has_pronoun,
            has_cardinal,
            has_modal
        ])

    def lex_featurize(self, comment):
        tokens = word_tokenize(comment)
        tokens = self.mark_negation(tokens)

        n_total = len(tokens)
        n_pos = 0
        n_neg = 0
        for t in tokens:

            if t.endswith('_NEG'):
                t = t.strip('_NEG')

                # Because this is negative,
                # add to opposite count.
                if t in self.pos_lex:
                    n_neg += 1
                elif t in self.neg_lex:
                    n_pos += 1

            else:
                if t in self.pos_lex:
                    n_pos += 1
                elif t in self.neg_lex:
                    n_neg += 1

        return np.array([
            n_pos - n_neg,
            n_pos / n_total,
            n_neg / n_total
        ])

    def mark_negation(self, tokens):
        """
        This is taken from Jeff Fossett's `NegationSuffixAdder`, found at
        <https://github.com/Fossj117/opinion-mining/blob/master/classes/transformers/tokenizers.py>

        His original documentation is below:

            Class to add simple negation marking to tokenized
            text to aid in sentiment analysis.

            A good explanation of negation marking, along with
            details of the approach used here can be found at:

            http://sentiment.christopherpotts.net/lingstruc.html#negation

            As defined in the link above, the basic approach is to
            "Append a _NEG suffix to every word appearing between a
            negation and a clause-level punctuation mark". Here, negation
            words are defined as those that match the NEGATION_RE regex, and
            clause-level punctuation marks are those that match the PUNCT_RE regex.

            Please note that this method is due to Das & Chen (2001) and
            Pang, Lee & Vaithyanathan (2002)

        """

        # negation tokenization
        neg_tokens = []
        append_neg = False # stores whether to add "_NEG"

        for token in tokens:

            # if we see clause-level punctuation,
            # stop appending suffix
            if self.PUNCT_RE.match(token):
                append_neg = False

            # Do or do not append suffix, depending
            # on state of 'append_neg'
            if append_neg:
                neg_tokens.append(token + "_NEG")
            else:
                neg_tokens.append(token)

            # if we see negation word,
            # start appending suffix
            if self.NEGATION_RE.match(token):
                append_neg = True

        return neg_tokens

