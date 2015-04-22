"""
For manipulating text.
"""

import string
from html.parser import HTMLParser

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet


class Vectorizer():
    def __init__(self, hash=False, min_df=0.015, max_df=0.9):
        """
        `min_df` is set to filter out extremely rare words,
        since we don't want those to dominate the distance metric.

        `max_df` is set to filter out extremely common words,
        since they don't convey much information.
        """

        if hash:
            args = [
                ('vectorizer', HashingVectorizer(input='content', stop_words='english', lowercase=True, tokenizer=Tokenizer())),
                ('tfidf', TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)),
                ('feature_reducer', TruncatedSVD(n_components=400)),
                ('normalizer', Normalizer(copy=False))
            ]
        else:
            args = [
                ('vectorizer', CountVectorizer(input='content', stop_words='english', lowercase=True, tokenizer=Tokenizer(), min_df=min_df, max_df=max_df)),
                ('tfidf', TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)),
                ('normalizer', Normalizer(copy=False))
            ]

        self.pipeline = Pipeline(args)

    def vectorize(self, docs, train=False):
        if train:
            return self.pipeline.fit_transform(docs)
        else:
            return self.pipeline.transform(docs)

    @property
    def vocabulary(self):
        return self.pipeline.named_steps['vectorizer'].get_feature_names()


class Tokenizer():
    """
    Custom tokenizer for vectorization.
    Uses Lemmatization.
    """
    def __init__(self):
        self.lemmr = WordNetLemmatizer()
        self.stops = stopwords.words('english')
        self.punct = {ord(p): ' ' for p in string.punctuation + '“”'}

        # Treat periods specially, replacing them with nothing.
        # This is so that initialisms like F.D.A. get preserved as FDA.
        self.period = {ord('.'): None}

    def __call__(self, doc):
        return self.tokenize(doc)

    def tokenize(self, doc):
        """
        Tokenizes a document, using a lemmatizer.

        Args:
            | doc (str)                 -- the text document to process.

        Returns:
            | list                      -- the list of tokens.
        """

        tokens = []

        # Strip punctuation.
        doc = doc.translate(self.period)
        doc = doc.translate(self.punct)

        for token in word_tokenize(doc):
            # Ignore punctuation and stopwords
            if token in self.stops:
                continue

            # Lemmatize
            lemma = self.lemmr.lemmatize(token.lower())
            tokens.append(lemma)

        return tokens


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)


def strip_tags(html):
    # Any unwrapped text is ignored,
    # so wrap html tags just in case.
    # Looking for a more reliable way of stripping HTML...
    html = '<div>{0}</div>'.format(html)
    s = MLStripper()
    s.feed(html)
    return s.get_data()


dash_map = {ord(p): ' ' for p in '—-'}
punct_map = {ord(p): '' for p in string.punctuation + '“”'}
def strip_punct(doc):
    return doc.translate(dash_map).translate(punct_map)


def html_decode(s):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    from: <http://stackoverflow.com/a/275246/1097920>
    """
    for code in (
            ("'", '&#39;'),
            ('"', '&quot;'),
            ('>', '&gt;'),
            ('<', '&lt;'),
            ('&', '&amp;')
        ):
        s = s.replace(code[1], code[0])
    return s





from gensim.models import Phrases
from nytnlp.keywords import rake
from textblob import Blobber
from textblob_aptagger import PerceptronTagger
blob = Blobber(pos_tagger=PerceptronTagger())
stops = stopwords.words('english')
lem = WordNetLemmatizer()
dash_map = {ord(p): ' ' for p in '—-'}
punct_map = {ord(p): '' for p in string.punctuation + '“”—’‘'}

# Trained on 100-200k NYT articles
bigram = Phrases.load('data/bigram_model.phrases')

def clean_doc(doc):
    doc = doc.lower()
    doc = doc.replace('\'s ', ' ')
    doc = doc.translate(dash_map)
    doc = doc.translate(punct_map)
    return doc


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
    phrases = [ph for ph in blo.noun_phrases + phrases if gram_size(ph) > 1]

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


def penn_to_wordnet(tag):
    """
    Convert a Penn Treebank PoS tag to WordNet PoS tag.
    """
    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        return wordnet.NOUN
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        return wordnet.VERB
    elif tag in ['RB', 'RBR', 'RBS']:
        return wordnet.ADV
    elif tag in ['JJ', 'JJR', 'JJS']:
        return wordnet.ADJ
    return None