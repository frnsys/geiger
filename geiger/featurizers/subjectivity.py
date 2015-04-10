import numpy as np
from textblob import Blobber
from textblob_aptagger import PerceptronTagger
from geiger.util.progress import Progress

class Featurizer():
    """
    Builds subjectivity features.

    Subjectivity lexicon sourced from
    <https://github.com/kuitang/Markovian-Sentiment/blob/master/data/subjclueslen1-HLTEMNLP05.tff>,
    presented in:

        Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005). Recognizing Contextual
        Polarity in Phrase-Level Sentiment Analysis. Proceedings of HLT/EMNLP 2005,
        Vancouver, Canada.

    This featurizer is largely based off Jeff Fossett's `SubjFeaturizer`, found at
    <https://github.com/Fossj117/opinion-mining/blob/master/classes/transformers/featurizers.py>
    """
    lex_path = 'data/subjclueslen1-HLTEMNLP05.tff'

    # Map NLTK POS tags to the lexicon's POS tags.
    TAG_MAP = {'NN': 'noun',
               'NNS': 'noun',
               'NNP': 'noun',
               'JJ': 'adj',
               'JJR': 'adj',
               'JJS': 'adj',
               'RB': 'adverb',
               'RBR': 'adverb',
               'RBS': 'adverb',
               'VB': 'verb',
               'VBD': 'verb',
               'VBG': 'verb',
               'VBN': 'verb',
               'VBP': 'verb',
               'VBZ': 'verb'}

    def __init__(self):
        self.lex = self.load_lexicon()
        self.blobber = Blobber(pos_tagger=PerceptronTagger())

    def load_lexicon(self):
        """
        Loads and processes the subjectivity lexicon.
        """
        lex = {}
        with open(self.lex_path, 'r') as f:
            for line in f.readlines():
                chunks = line.strip().split(' ')
                data = dict([c.split('=') for c in chunks if '=' in c])
                lex[(data['word1'], data['pos1'])] = {
                    'subjectivity': 1 if data['type'] == 'strongsubj' else 0,

                    # TO DO should polarity be included as a feature?
                    'priorpolarity': data['priorpolarity']
                }
        return lex

    def featurize(self, comments, return_ctx=False):
        #feats = np.vstack([self._featurize(c.body) for c in comments])

        p = Progress('SUBJ')
        n = len(comments)

        feats = []
        for i, c in enumerate(comments):
            p.print_progress((i+1)/n)
            feats.append(self._featurize(c.body))
        feats = np.vstack(feats)

        if return_ctx:
            return feats, feats
        else:
            return feats

    def _featurize(self, text):
        """
        Featurize a single document.
        """
        tagged = self.blobber(text).tags

        n_strong = 0
        n_weak = 0
        n_total = len(tagged)

        for w, tag in tagged:
            # Map the NLTK pos tag to the lexicon's pos tags.
            pos = self.TAG_MAP[tag] if tag in self.TAG_MAP else None

            try:
                info = self.lex[(w, pos)]
            except KeyError:
                try:
                    info = self.lex[(w, 'anypos')]
                except KeyError:
                    continue

            if info['subjectivity'] == 1:
                n_strong += 1
            else:
                n_weak += 1

        return np.array([
            n_strong/n_total,
            n_weak/n_total,
            n_strong + n_weak
        ])
