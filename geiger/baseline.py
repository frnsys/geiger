import string
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from geiger.clusters import cluster


def baseline(docs, eps=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], metric='euclidean'):
    dist_mat = _compute_dist_mat(docs, metric)
    clusters = cluster(dist_mat, eps)
    return [[docs[i] for i in clus] for clus in clusters]


def _compute_dist_mat(docs, metric):
    v = Vectorizer()
    vecs = v.vectorize(docs, train=True)
    dist_mat = pdist(vecs.todense(), metric=metric)
    return squareform(dist_mat)


class Vectorizer():
    def __init__(self, min_df=0.015, max_df=0.9):
        """
        `min_df` is set to filter out extremely rare words,
        since we don't want those to dominate the distance metric.

        `max_df` is set to filter out extremely common words,
        since they don't convey much information.
        """

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
