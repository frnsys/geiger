comments_path = 'data/comments.csv'

# Clusters must be above this size to be included
min_cluster_size = 5

# Clusters consist of comments which are closer to each other than this distance
distance_cutoff = 0.5

# Clustering params
lower_limit_scale = 0.9
upper_limit_scale = 1.2

# Auth info
scoop_auth = ('(-__-)', '(-__-)')
scoop_base = '(-__-)'

community_key = '(-__-)'
community_base = '(-__-)'

# Params for featurizers
featurizers = {
    'bow': {},
    'keyword': {},
    'subjectivity': {},
    'opinion': {},
    'doc2vec': {},
    'polisent': {},
    'topics': {
        'n_topics': 10
    }
}

# Where trained models are stored
models_path = 'data/'
d2v_path = 'data/comments.doc2vec'

# Whether or not sentences should be the object of examination.
# If True, treats each sentence as an independent document.
# If False, operates on comments as a whole (but still returns highlights as
# sentences).
sentences = False
