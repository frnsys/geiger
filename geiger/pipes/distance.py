from broca import Pipe
import scipy.sparse as sps
from scipy.spatial.distance import pdist, squareform


class Distance(Pipe):
    input = Pipe.type.vecs
    output = Pipe.type.dist_mat

    def __init__(self, metric='euclidean'):
        self.metric = metric

    def __call__(self, vecs):
        if sps.issparse(vecs):
            vecs = vecs.todense()
        dist_mat = pdist(vecs, metric=self.metric)
        return squareform(dist_mat)
