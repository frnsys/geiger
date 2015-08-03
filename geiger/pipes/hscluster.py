from broca import Pipe
from broca.common.util import dist_to_sim
from hscluster import hscluster


class HSCluster(Pipe):
    input = Pipe.type.dist_mat
    output = Pipe.type.clusters

    def __call__(self, dist_mat):
        """
        Returns clusters as a list of document indices.
        """
        sim_mat = dist_to_sim(dist_mat)
        labels = hscluster(sim_mat)
        n = max(labels) + 1
        clusters = [[] for _ in range(n)]
        for i in range(len(labels)):
            if labels[i] >= 0:
                clusters[labels[i]].append(i)
        return clusters
