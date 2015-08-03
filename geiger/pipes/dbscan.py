from broca import Pipe
from sklearn import cluster
from broca.cluster.parameter import estimate_eps


class DBSCAN(Pipe):
    input = Pipe.type.dist_mat
    output = Pipe.type.clusters

    def __init__(self, eps=None, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples

    def __call__(self, dist_mat):
        """
        Returns clusters as a list of document indices.
        """
        if self.eps is None:
            self.eps = estimate_eps(dist_mat)[0]

        self.m = cluster.DBSCAN(metric='precomputed', eps=self.eps, min_samples=self.min_samples)
        labels = self.m.fit_predict(dist_mat)
        n = max(labels) + 1

        if n == 0:
            return []

        else:
            # Outliers (with label=-1) are not returned
            clusters = [[] for _ in range(n)]
            for i in range(len(labels)):
                if labels[i] >= 0:
                    clusters[labels[i]].append(i)
            return clusters
