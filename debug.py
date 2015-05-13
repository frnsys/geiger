import os
import math
import json
import shutil
import numpy as np
from time import time
from collections import defaultdict
from geiger.semsim import SemSim, idf


if __name__ == '__main__':
    #path_to_examples = 'data/examples/climate_example.json'
    #path_to_examples = 'data/examples/clinton_example.json'
    path_to_examples = 'data/examples/gaymarriage_example.json'
    data = json.load(open(path_to_examples, 'r'))
    docs = [d['body'] for d in data if len(d['body']) > 140]

    # Remove duplicates
    docs = list(set(docs))

    outdir = '_debug'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    t0 = time()
    semmy = SemSim(debug=True)
    results, all_clusters = semmy.cluster(docs, eps=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    elapsed = time() - t0
    print('------done in %.2fs------' % (elapsed))

    for e, clusters in results.items():
        print('-------------------------------------------------')
        print('dbscan with eps={0}'.format(e))
        print('num clusters:')
        print(len(clusters))
        print('num comments:')
        print(sum([len(clus) for clus in clusters]))
        print('cluster sizes:')
        print([len(clus) for clus in clusters])
        print('cluster score:')
        print(semmy._score_clusters(clusters, len(docs)))

        eps_dir = os.path.join(outdir, 'eps_{0}'.format(e))
        os.makedirs(eps_dir)
        for i, clus in enumerate(clusters):
            clus_dir = os.path.join(eps_dir, 'clus_{0}'.format(i))
            os.makedirs(clus_dir)

            kw_sets = []
            for j, (idx, c) in enumerate(clus):
                kw_sets.append(set(semmy.docs[idx]))
                outfile = os.path.join(clus_dir, '{0}.txt'.format(j))
                with open(outfile, 'w', encoding='utf-8') as f:
                    f.write('\n\n'.join([c,str(set(semmy.docs[idx]))]))

            all_kw_counts = defaultdict(int)
            for kws in kw_sets:
                for kw in kws:
                    all_kw_counts[kw] += 1
            ranked_kws = ['{0} [{1}]'.format(kw, all_kw_counts[kw] * semmy.saliences[kw]) for kw in sorted(all_kw_counts.keys(), key=lambda k: all_kw_counts[k], reverse=True)]
            outfile = os.path.join(clus_dir, '_keywords.txt')
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write('{0}\n-------------\n\n{1}'.format('\n'.join(ranked_kws[:10]), '\n'.join(ranked_kws[10:])))

    all_clusters_dir = os.path.join(outdir, '_all_clusters')
    os.makedirs(all_clusters_dir)
    for i, clus in enumerate(all_clusters):
        clus_dir = os.path.join(all_clusters_dir, 'clus_{0}'.format(i))
        os.makedirs(clus_dir)

        kw_sets = []
        for j, (idx, c) in enumerate(clus):
            kw_sets.append(set(semmy.docs[idx]))
            outfile = os.path.join(clus_dir, '{0}.txt'.format(j))
            with open(outfile, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join([c,str(set(semmy.docs[idx]))]))

        all_kw_counts = defaultdict(int)
        for kws in kw_sets:
            for kw in kws:
                all_kw_counts[kw] += 1
        ranked_kws = ['{0} [{1}]'.format(kw, all_kw_counts[kw] * semmy.saliences[kw]) for kw in sorted(all_kw_counts.keys(), key=lambda k: all_kw_counts[k], reverse=True)]
        outfile = os.path.join(clus_dir, '_keywords.txt')
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write('{0}\n-------------\n\n{1}'.format('\n'.join(ranked_kws[:10]), '\n'.join(ranked_kws[10:])))

    for i, doc in enumerate(docs):
        kws = semmy.docs[i]

        idf_scored_kws = [(kw, idf.get(kw, 0)) for kw in set(kws)]
        idf_sorted_scored_kws = sorted(idf_scored_kws, key=lambda x: x[1], reverse=True)

        iidf_scored_kws = [(kw, semmy.iidf.get(kw, 0)) for kw in set(kws)]
        iidf_sorted_scored_kws = sorted(iidf_scored_kws, key=lambda x: x[1], reverse=True)

        lidf_scored_kws = [(kw, semmy.saliences[kw]) for kw in set(kws)]
        lidf_sorted_scored_kws = sorted(lidf_scored_kws, key=lambda x: x[1], reverse=True)

        kw_support = defaultdict(int)
        for kw in kws:
            kw_support[kw] += 1

        tfidf_scored_kws = [(kw, (1 + math.log(kw_support[kw])) * idf.get(kw, 0)) for kw in set(kws)]
        tfidf_sorted_scored_kws = sorted(tfidf_scored_kws, key=lambda x: x[1], reverse=True)

        tfiidf_scored_kws = [(kw, (1 + math.log(kw_support[kw])) * semmy.iidf.get(kw, 0)) for kw in set(kws)]
        tfiidf_sorted_scored_kws = sorted(tfiidf_scored_kws, key=lambda x: x[1], reverse=True)

        tflidf_scored_kws = [(kw, (1 + math.log(kw_support[kw])) * semmy.saliences[kw]) for kw in set(kws)]
        tflidf_sorted_scored_kws = sorted(tflidf_scored_kws, key=lambda x: x[1], reverse=True)

        # most similar
        top_indices = np.argpartition(semmy.sim_mat[i], -5)[-5:]
        sims = semmy.sim_mat[i,top_indices]
        sim_docs = []
        for j, s in enumerate(sims):
            sim_docs.append('sim: {0}\n{1}\n{2}'.format(s, docs[j], j))

        with open(os.path.join(outdir, '{0}.txt'.format(i)), 'w') as f:
            f.write('{0}\n\nmost similar:\n{7}\n\nidf ranked keywords:\n{1}\n\niidf ranked keywords:\n{2}\n\nlidf ranked keywords:\n{3}\n\ntfidf ranked keywords:\n{4}\n\ntfiidf ranked keywords:\n{5}\n\ntflidf ranked keywords:\n{6}\n\nmost similar:\n{7}'.format(
                doc,
                '\n'.join([str(s) for s in idf_sorted_scored_kws]),
                '\n'.join([str(s) for s in iidf_sorted_scored_kws]),
                '\n'.join([str(s) for s in lidf_sorted_scored_kws]),
                '\n'.join([str(s) for s in tfidf_sorted_scored_kws]),
                '\n'.join([str(s) for s in tfiidf_sorted_scored_kws]),
                '\n'.join([str(s) for s in tflidf_sorted_scored_kws]),
                '\n\n'.join(sim_docs)
            ))

    with open(os.path.join(outdir, '_keywords.txt'), 'w') as f:
        output = []
        all_terms = sorted(list(semmy.all_terms_unfiltered), key=lambda k: semmy.saliences[k], reverse=True)
        for kw in all_terms:
            output.append('{0}\t\t{1}\t{2}\t{3}'.format(kw,
                                                             semmy.saliences[kw],
                                                             idf.get(kw, 0),
                                                             semmy.iidf.get(kw, 0)))
        f.write('\n'.join(output))


    with open(os.path.join(outdir, '_keywords_filtered.txt'), 'w') as f:
        output = []
        all_terms = sorted(list(semmy.all_terms), key=lambda k: semmy.saliences[k], reverse=True)
        for kw in all_terms:
            output.append('{0}\t\t{1}\t{2}\t{3}'.format(kw,
                                                             semmy.saliences[kw],
                                                             idf.get(kw, 0),
                                                             semmy.iidf.get(kw, 0)))
        f.write('\n'.join(output))


