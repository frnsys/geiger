"""
Handles vectorizing of documents.
"""

import string

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


class Vectorizer():
    def __init__(self, hash=False):
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
                ('vectorizer', CountVectorizer(input='content', stop_words='english', lowercase=True, tokenizer=Tokenizer(), min_df=0.015, max_df=0.9)),
                ('tfidf', TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)),
                ('normalizer', Normalizer(copy=False))
            ]

        self.pipeline = Pipeline(args)

    def vectorize(self, docs, train=False):
        if train:
            return self.pipeline.fit_transform(docs)
        else:
            return self.pipeline.transform(docs)


class Tokenizer():
    """
    Custom tokenizer for vectorization.
    Uses Lemmatization.
    """
    def __init__(self):
        self.lemmr = WordNetLemmatizer()

    def __call__(self, doc):
        return self.tokenize(doc, lemmr=self.lemmr)

    def tokenize(self, doc, **kwargs):
        """
        Tokenizes a document, using a lemmatizer.

        Args:
            | doc (str)                 -- the text document to process.

        Returns:
            | list                      -- the list of tokens.
        """

        tokens = []
        lemmr = kwargs.get('lemmr', WordNetLemmatizer())
        stops = set(list(string.punctuation) + stopwords.words('english'))

        # Tokenize
        for sentence in sent_tokenize(doc):
            for token in word_tokenize(sentence):

                # Ignore punctuation and stopwords
                if token in stops:
                    continue

                # Lemmatize
                lemma = lemmr.lemmatize(token.lower())
                tokens.append(lemma)
        return tokens



from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def html_decode(s):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    from: <http://stackoverflow.com/a/275246/1097920
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