from itertools import product
from collections import defaultdict
from geiger.semsim import SemSim
from geiger.text.tokenize import blob, lem
from geiger.clusters import estimate_eps
from nytnlp.util import penn_to_wordnet


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

        # Lemmatize kws
        keywords = []
        for i in ids:
            kws = []
            for kw in map[i]['keywords']:
                tag = blob(kw).tags[0][1]
                tag = penn_to_wordnet(tag)
                if tag is not None:
                    kws.append(lem.lemmatize(kw, tag))
                else:
                    kws.append(kw)
            keywords.append(kws)

        # Some comments may belong to more than one cluster,
        # We create permutations of every possible list of labels
        all_labels = list(product(*raw_labels))

        return bodies, keywords, all_labels, raw_labels


def _evaluate_keywords(docs, keywords, verbose=True):
    """
    Check keyword tokenization against annotated keywords
    """
    s = SemSim(debug=True)
    results = []
    for i, toks in enumerate(s._preprocess(docs)):
        anno = set(keywords[i])
        toks = set([t.term for t in toks])
        inter = len(anno & toks)
        found = inter/len(anno)
        extra = toks - (anno & toks)
        missing = anno - (anno & toks)

        if verbose:
            print('----------------')
            print('{} annotated keywords found in tokens'.format(found))
            print('Missing: {}'.format(missing))
            print('{} extra keywords found in tokens'.format(extra))

        results.append({
            'true': anno,
            'predicted': toks,
            'missing': missing,
            'extra': extra,
            'p_found': found
        })
    return results


def evaluate(truth_file, verbose=True):
    bodies, kws, all_true_labels, raw_true_labels = parse_clusters(truth_file)

    kw_results = _evaluate_keywords(bodies, kws, verbose=verbose)

    # Check clustering performance
    s = SemSim()
    clus_results = []
    clusters, descriptors = s.cluster(bodies)
                                      #eps=[3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                           #4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,
                                           #5., 5.1, 5.2, 5.3, 5.5, 5.5, 5.6, 5.7, 5.8, 5.9])
                                      #eps=[0.5, 0.6, 0.7, 0.8, 0.9,
                                           #1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                           #2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                           #3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                                           #4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9])


    # Convert clusters into labels for each comment
    raw_pred_labels = [[] for i in range(len(bodies))]
    for i, clus in enumerate(clusters):
        for doc in clus:
            raw_pred_labels[doc.id].append(i)

    raw_pred_labels = [lbls if lbls else [-1] for lbls in raw_pred_labels]
    print('PREDICTED LABELS:')
    print(raw_pred_labels)

    # Find correct and incorrect co-cluster pairings
    classification_results = []
    for i, doc in enumerate(bodies):
        fp, fn, tp, tn = 0, 0, 0, 0
        n_tp, n_tn = 0, 0
        for j, doc_ in enumerate(bodies):
            if i == j:
                continue

            # Check true
            true_co_cluster = False
            if set(raw_true_labels[i]).intersection(set(raw_true_labels[j])):
                true_co_cluster = True
                n_tp += 1
            else:
                n_tn += 1

            # Check predicted
            pred_co_cluster = False
            intersect = set(raw_pred_labels[i]).intersection(set(raw_pred_labels[j]))
            if intersect and intersect != {-1}:
                pred_co_cluster = True

            if true_co_cluster != pred_co_cluster:
                if true_co_cluster:
                    fn += 1
                else:
                    fp += 1
            else:
                if true_co_cluster:
                    tp += 1
                else:
                    tn += 1
        classification_results.append({
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'tn': tn,
            'tpr': tp/n_tp if n_tp else 1,
            'tnr': tn/n_tn if n_tn else 1
        })

    summary = {
        'avg_p_found': sum(r['p_found'] for r in kw_results)/len(kw_results),
        'avg_extra': sum(len(r['extra']) for r in kw_results)/len(kw_results),
        'avg_fp': sum(r['fp'] for r in classification_results)/len(classification_results),
        'avg_fn': sum(r['fn'] for r in classification_results)/len(classification_results),
        'avg_tpr': sum(r['tpr'] for r in classification_results)/len(classification_results),
        'avg_tnr': sum(r['tnr'] for r in classification_results)/len(classification_results),
        'clf_results': classification_results
    }

    # Show max-sim pairs across _all_ terms
    s._all_max_sim_pairs()

    #print('----------------------')
    #print('VECTOR REPRESENTATIONS')
    #print('----------------------')
    #vecs = s._vec_reps()
    #print('----------------------')
    #print('----------------------')

    #from sklearn.cluster import MeanShift
    #print('Mean shift clustering...')
    #for bw in [0.2, 0.4, 0.8, 1.6, 2., 2.4, 2.6, 3.2]:
        #print('bandwidth: {}'.format(bw))
        #ms = MeanShift(cluster_all=False, bandwidth=bw)
        #ms_labels = ms.fit_predict(vecs)
        #print(ms_labels)

    #from sklearn.cluster import DBSCAN
    #print('DBSCAN clustering...')
    #for eps in [0.2, 0.4, 0.8, 1.2, 1.6, 2., 2.4, 2.6, 3.2, 4., 5., 6., 10., 20.]:
        #print('eps: {}'.format(eps))
        #db = DBSCAN(eps=eps, min_samples=2)
        #db_labels = db.fit_predict(vecs)
        #print(db_labels)

    # To try and estimate a value for eps
    dm = s.dist_mat.copy()
    estimates = estimate_eps(dm)
    print('ESTIMATES:')
    print(estimates)

    return kw_results, clus_results, s.docs, s.all_terms, s.pruned, all_true_labels, raw_pred_labels, summary
