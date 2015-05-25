from itertools import product
from collections import defaultdict
from sklearn import metrics
from geiger.semsim import SemSim


def parse_clusters(fname):
    with open(fname, 'r') as f:
        clusters = ['']
        for line in f.readlines():
            if line.strip() == '---':
                clusters.append('')
                continue
            clusters[-1] += line

        # Do a lot of wrangling (maybe too much)
        ids = set()
        map = defaultdict(lambda: {'labels':[]}) # map comment ids to data
        for i, cluster in enumerate(clusters):
            parts = [p for p in cluster.split('\n') if p]
            # Iterate as triples
            for id, body, keywords in zip(*[iter(parts)]*3):
                id = int(id.replace('ID:', ''))
                ids.add(id)
                map[id]['labels'].append(i)
                map[id]['body'] = body
                map[id]['keywords'] = [kw.replace('_', ' ') for kw in keywords.lower().split(' ')]

        ids = list(ids)
        raw_labels = [map[i]['labels'] for i in ids]
        bodies = [map[i]['body'] for i in ids]
        keywords = [map[i]['keywords'] for i in ids]

        # Some comments may belong to more than one cluster,
        # We create permutations of every possible list of labels
        all_labels = list(product(*raw_labels))

        return bodies, keywords, all_labels


def most_similar(cluster, clusters):
    """
    Find the most similar cluster for a given cluster (both represented as labels)
    """
    return max(clusters, key=lambda c: set(cluster) & set(c))


def evaluate(truth_file):
    bodies, kws, all_true_labels = parse_clusters(truth_file)

    # Check keyword tokenization against annotated keywords
    s = SemSim(debug=True)
    kw_results = []
    for i, toks in enumerate(s._preprocess(bodies)):
        anno = set(kws[i])
        toks = set([t.term for t in toks])
        inter = len(anno & toks)
        found = inter/len(anno)
        extra = toks - (anno & toks)
        missing = anno - (anno & toks)
        print('----------------')
        print('{} annotated keywords found in tokens'.format(found))
        print('Missing: {}'.format(missing))
        print('{} extra keywords found in tokens'.format(extra))

        kw_results.append({
            'true': anno,
            'predicted': toks,
            'missing': missing,
            'extra': extra,
            'p_found': found
        })


    # Check clustering performance
    s = SemSim()
    clus_results = []
    clusters, descriptors = s.cluster(bodies,
                                      eps=[0.5, 0.6, 0.7, 0.8, 0.9,
                                           1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                           2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                           3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9])


    # Convert clusters into labels for each comment
    raw_pred_labels = [[] for i in range(len(bodies))]
    for i, clus in enumerate(clusters):
        for doc in clus:
            raw_pred_labels[doc.id].append(i)

    raw_pred_labels = [lbls if lbls else [-1] for lbls in raw_pred_labels]

    # We don't create all label permutations because there are so damn many
    maxl = max(len(l) for l in raw_pred_labels) - 1
    pred_labels = [[labels[min(l, len(labels) - 1)] for labels in raw_pred_labels] for l in range(maxl)]
    for labels in pred_labels:
        for labels_true in all_true_labels:
            scores = {
                'adjusted_rand': metrics.adjusted_rand_score(labels_true, labels),
                'adjusted_mutual_info': metrics.adjusted_mutual_info_score(labels_true, labels),
                'homogeneity': metrics.homogeneity_score(labels_true, labels),
                'completeness': metrics.completeness_score(labels_true, labels)
            }
            if scores['adjusted_mutual_info'] <= 0 or scores['adjusted_rand'] <= 0:
                continue
            print('--------')
            true_clusters = defaultdict(list)
            for i, label in enumerate(labels_true):
                true_clusters[label].append(i)
            pred_clusters = defaultdict(list)
            for i, label in enumerate(labels):
                pred_clusters[label].append(i)
            print(list(true_clusters.values()))
            print(list(pred_clusters.values()))

            for clus in true_clusters.values():
                print('\t{}'.format(clus))
                print('\t{}'.format(most_similar(clus, pred_clusters.values())))
                print('\t---')

            print(scores)
            print('--------')

            clus_results.append({
                'scores': scores
            })

    return kw_results, clus_results, s.docs, s.all_terms, s.pruned, all_true_labels
