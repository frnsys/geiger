import math
import numpy as np
from broca import Pipe
from broca.common.util import gram_size
from collections import Counter, defaultdict
from geiger.knowledge import W2V, IDF
from sup.progress import Progress

import config
w2v = W2V(remote=config.remote)
idf = IDF(remote=config.remote)


class Doc():
    def __init__(self, id, terms):
        self.id = id
        self.terms = sorted(terms, key=lambda t: t.salience, reverse=True)
        self.term_freqs = {t: f for t, f in list(Counter(self.terms).items())}

        # Keep track of max-sim pairs and similarities against each other document
        # (for debugging)
        self.pairs = {}
        self.sims = {}

    def __iter__(self):
        return iter(self.terms)

    def __contains__(self, term):
        return term in self.terms

    def __getitem__(self, i):
        return self.terms[i]

    def to_json(self):
        return {
            'id': self.id,
            'terms': [t.to_json() for t in self.terms],
            'terms_uniq': [t.to_json() for t in set(self.terms)],
            'term_freqs': {t.term: f for t, f in self.term_freqs.items()},
            'pairs': {d.id: [(t1.term, t2.term, sim) for t1, t2, sim in pairs] for d, pairs in self.pairs.items()},
            'sims': {d.id: sim for d, sim in self.sims.items()},
            'dists': {d.id: 1/sim - 1 if sim else 9999999. for d, sim in self.sims.items()},
            'highlighted': getattr(self, 'highlighted', None)
        }


class Term():
    def __init__(self, term, salience, internal_idf, global_idf):
        self.term = term
        self.salience = salience
        self.iidf = internal_idf
        self.gidf = global_idf

    def __repr__(self):
        return '<{}{}{} [S:{:.2f}|L:{:.2f}|G:{:.2f}]>'.format(
            '\033[92m',
            self.term,
            '\033[0m',
            self.salience,
            self.iidf,
            self.gidf)

    def __eq__(self, other):
        return self.term == other.term

    def __hash__(self):
        return hash(self.term)

    def __contains__(self, term):
        return term.term in self.term

    def to_json(self):
        return self.__dict__


class SemSim(Pipe):
    """
    Clusters tokenized documents by semantic similarity.

    A "term" is a keyword or a keyphrase.
    """
    input = Pipe.type.tokens
    output = Pipe.type.dist_mat

    def __init__(self, debug=False, min_salience=0.2, idf_as_salience=False):
        self.debug = debug
        self.min_salience = min_salience

        # You can set this to be true if you want to use avg local/global idf
        # instead of salience (for testing purposes)
        self.idf_as_salience = idf_as_salience


    def __call__(self, token_docs):
        filtered_token_docs = []
        for doc in token_docs:
            # Remove keyphrases with more than 3 words to reduce runtime
            filtered_token_docs.append([t for t in doc if gram_size(t) <= 3])
        token_docs = self._preprocess(filtered_token_docs)
        dist_mat = self._distance_matrix(token_docs)
        return dist_mat

        # Build descriptors for each cluster
        # TO DO clean this up
        #descriptors = []
        #for clus in clusters:
            #all_term_counts = defaultdict(int)
            #for doc in clus:
                #for term in set(doc.terms):
                    #all_term_counts[term] += 1
            #ranked_terms = [(term, all_term_counts[term] * term.salience) for term in sorted(all_term_counts.keys(), key=lambda t: all_term_counts[t] * t.salience, reverse=True)]
            #descriptors.append(ranked_terms)

        #return clusters, descriptors


    def _sim_weak(self, d1, d2):
        """
        Compute similarity based on ratio of exactly overlapping terms, weighted by
        each terms' salience.
        """
        d1 = set(d1)
        d2 = set(d2)

        kw_i = sum(self.saliences[kw] for kw in d1.intersection(d2))
        kw_u = sum(self.saliences[kw] for kw in d1.union(d2))

        return kw_i/kw_u


    def _sim_strong(self, d1, d2):
        """
        Like weak similarity, but penalizes for non-overlapping terms, weighted
        by their salience.
        """
        d1 = set(d1)
        d2 = set(d2)

        kw_i = sum(self.saliences[kw] for kw in d1.intersection(d2))
        kw_d = sum(self.saliences[kw] for kw in d1.symmetric_difference(d2))
        kw_u = sum(self.saliences[kw] for kw in d1.union(d2))

        return (kw_i - kw_d)/(2*kw_u) + 1/2


    def _sim_sem(self, d1, d2):
        """
        Like weak similarity, but does not require exactly overlapping terms,
        just the most similar terms.
        """
        pairs = self._semsim_pairs(d1, d2)

        if not pairs:
            return 0

        sims, sals = zip(*[(sim, (t1.salience + t2.salience)/2) for t1, t2, sim, in pairs])
        #sims, sals = zip(*[(sim, (t1.salience + t2.salience)/2) if sim > 0 else (sim, min(t1.salience, t2.salience)) for t1, t2, sim, in pairs])

        sim = sum(sim * sal for sim, sal in zip(sims, sals))/sum(sals)

        # For debugging
        d1.sims[d2] = sim
        d2.sims[d1] = sim

        return sim


    def _term_sim(self, t1, t2):
        """
        Get word2vec similarity for two terms.
        """
        t1 = t1.term
        t2 = t2.term

        if t1 == t2:
            return 1.

        list_comp = False

        # Convert to forms present in the w2v model
        t1 = t1.replace(' ', '_')
        if t1 not in w2v.vocab:
            list_comp = True
            t1 = t1.split('_')

        t2 = t2.replace(' ' , '_')
        if t2 not in w2v.vocab:
            t2 = t2.split('_')
            if not list_comp:
                t1 = [t1]
            list_comp = True
        elif list_comp:
            t2 = [t2]

        try:
            if list_comp:
                sim = w2v.n_similarity(t1, t2)
            else:
                sim = w2v.similarity(t1, t2)

            if sim <= 0.4:
                sim = 0

        # Word not in vocab
        except KeyError:
            sim = 0

        return sim


    def _similarity_matrix(self, docs):
        """
        Compute the full similarity matrix for some documents
        """
        if self.debug:
            print('building similarity matrix...')

        p = Progress()
        n = len(docs)
        sim_mat = np.zeros((n, n))

        n = n**2
        n = n/2

        # not efficient, w/e this is a sketch
        count = 0
        for i, d1 in enumerate(docs):
            for j, d2 in enumerate(docs):
                if i == j:
                    sim_mat[i,j] = 1.
                    count += 1
                    p.print_progress(count/n)

                # Just build the lower triangle
                elif i > j:
                    sim_mat[i,j] = self._sim_sem(d1, d2)
                    count += 1
                    p.print_progress(count/n)

        if self.debug:
            print('done building similiarty matrix.')

        # Construct the full sim mat from the lower triangle
        return sim_mat + sim_mat.T - np.diag(sim_mat.diagonal())


    def _internal_idf(self, docs):
        """
        Compute intra-comment IDF
        """
        N = len(docs)
        iidf = defaultdict(int)
        for terms in docs:
            # Only care about presence, not frequency,
            # so convert to a set
            for t in set(terms):
                iidf[t] += 1

        for k, v in iidf.items():
            iidf[k] = math.log(N/v + 1)

        # Normalize
        mxm = max(iidf.values())
        for k, v in iidf.items():
            iidf[k] = v/mxm

        return iidf


    def _salience(self, t):
        """
        Compute the salience of a term
        """

        idf_c = self.iidf[t]
        sal_c = self._norm(idf_c)

        idf_g = idf.get(t, idf_c)
        sal_g = self._norm(idf_g)

        if self.idf_as_salience:
            return (idf_c + idf_g)/2

        return (sal_c + sal_g)/2


    def _norm(self, x):
        n = (x-0.5)**2
        return math.exp(-n/0.05)


    def _semsim_pairs(self, d1, d2):
        """
        Construct maximally-semantically-similar pairs b/w terms of two documents

        Pairs are returned with the similarities so they don't need to be re-computed.
        """
        pairs1 = set()
        pairs2 = set()

        if not d1.terms or not d2.terms:
            return set()

        # Extract sub-similarity-matrix for the terms here
        rows = [[self.w2v_term_map[t]] for t in d1]
        cols = [self.w2v_term_map[t] for t in d2]
        sub_mat = self.w2v_sim_mat[rows, cols]

        # Compute necessary similarities
        uncomputed = np.where(sub_mat == -1)
        for i, j in zip(*uncomputed):
            sub_mat[i,j] = self._term_sim(d1[i], d2[j])

        # Update the main similarity matrix
        self.w2v_sim_mat[rows,cols] = sub_mat

        rows = [r[0] for r in rows]
        cols = [[c] for c in cols]
        self.w2v_sim_mat[cols,rows] = sub_mat.T

        # Create max-sim pairs
        # Max-sim pairs for d1->d2
        for i, j in enumerate(np.argmax(sub_mat, axis=1)):
            pairs1.add((d1[i], d2[j], sub_mat[i,j]))
        d1.pairs[d2] = pairs1

        # Max-sim pairs for d2->d1
        for j, i in enumerate(np.argmax(sub_mat, axis=0)):
            # Add in this order so we recognize duplicate pairs
            pairs2.add((d1[i], d2[j], sub_mat[i,j]))
        d2.pairs[d1] = pairs2

        return pairs1.union(pairs2)


    def _prune(self, docs):
        """
        Aggressively prune noisy terms:
            - those that appear only in one document (IDF is 1.0)
            - those that are not sufficiently salient
        This improves runtime and should improve output quality
        """
        for doc in docs:
            original_terms = set(doc.terms)
            if self.idf_as_salience:
                doc.terms = [t for t in doc if t.salience >= self.min_salience and t.salience <= 0.9]
            else:
                doc.terms = [t for t in doc if t.salience >= self.min_salience]

            # See what terms were removed
            pruned = original_terms.difference(set(doc.terms))

        print('Pruned:')
        print(pruned)
        return docs, pruned


    def _preprocess(self, token_docs):
        if self.debug:
            print('clustering {0} docs'.format(len(token_docs)))

        # Compute intra-comment IDF and salience for all terms
        self.iidf = self._internal_idf(token_docs)
        self.all_terms = {t for terms in token_docs for t in terms}
        self.saliences = {t: self._salience(t) for t in self.all_terms}

        # Proper representations
        # Keep it so that each term only has one Term instance
        # TO DO clean this up
        self.all_terms = {Term(t, self.saliences[t], self.iidf[t], idf[t]) for t in self.all_terms}
        term_map = {t.term: t for t in self.all_terms}
        docs = [Doc(i, [term_map[t] for t in doc]) for i, doc in enumerate(token_docs)]

        # testing
        self.all_terms_unfiltered = self.all_terms

        docs, self.pruned = self._prune(docs)
        self.all_terms = {t for terms in docs for t in terms}

        # Compute normalized saliences
        self.normalized_saliences = {}
        mxm = max(self.saliences.values())
        for k, v in self.saliences.items():
            self.normalized_saliences[k] = v/mxm
        for t in self.all_terms:
            t.normalized_salience = self.normalized_saliences[t.term]

        if self.debug:
            print('vocabulary has {0} terms'.format(len(self.all_terms)))

        return docs


    def _distance_matrix(self, tokens):
        # Cache a w2v sim mat for faster lookup
        n = len(self.all_terms)
        self.w2v_sim_mat = np.full((n, n), -1)

        # Map terms to their indices in the w2v sim mat
        self.w2v_term_map = {t: i for i, t in enumerate(self.all_terms)}

        # Compute similarity matrix and convert to a distance matrix
        sim_mat = self._similarity_matrix(tokens)
        sim_mat[np.where(sim_mat == 0)] = 0.000001
        dist_mat = 1/sim_mat - 1
        self.sim_mat = sim_mat
        self.dist_mat = dist_mat
        return dist_mat

    def _all_max_sim_pairs(self):
        """
        Show max-sim pairs across _all_ terms.
        Requires that the w2v sim mat is already computed

        (mainly for debugging)
        """
        tsimmat = self.w2v_sim_mat.copy()
        tsimmat[np.where(tsimmat == 1.)] = -1
        for term in self.all_terms:
            idx = self.w2v_term_map[term]
            top = np.nanargmax(tsimmat[idx])
            sim = np.nanmax(tsimmat[idx])
            # bleh, not efficient
            for k, v in self.w2v_term_map.items():
                if v == top:
                    match = k
                    break
            print('({}, {}, {})'.format(term, match, sim))


    def _vec_reps(self):
        """
        Creates salience-weighted vector representations for documents
        """

        # Keep track of which term pairs collapse
        self.collapse_map = {}

        # Identify which terms to collapse
        tsimmat = self.w2v_sim_mat.copy()
        tsimmat[np.where(tsimmat == 1.)] = -1
        for term in self.all_terms:
            idx = self.w2v_term_map[term]
            top = np.nanargmax(tsimmat[idx])
            sim = np.nanmax(tsimmat[idx])

            if sim >= 0.8: # cutoff
                # bleh, find matching term by index
                for k, v in self.w2v_term_map.items():
                    if v == top:
                        match = k
                        break

                # Only collapse terms of the same gram size
                # This is because phrases which share a word in common tend to have
                # a higher similarity, because they share a word in common
                # TO DO could collapse terms of diff gram sizes but require a higher
                # sim threshold
                if gram_size(term.term) == gram_size(match.term):
                    # If either term is already in the collapse map
                    if term in self.collapse_map:
                        self.collapse_map[match] = self.collapse_map[term]
                    elif match in self.collapse_map:
                        self.collapse_map[term] = self.collapse_map[match]
                    else:
                        self.collapse_map[term] = term
                        self.collapse_map[match] = term

        # Build the reduced term set
        self.collapsed_terms = set()
        for term in self.all_terms:
            self.collapsed_terms.add(self.collapse_map.get(term, term))

        print(len(self.all_terms))
        print(len(self.collapsed_terms))

        terms = list(self.collapsed_terms)

        # Now we can build the vectors
        # TO DO make this not ridiculous
        vecs = []
        for d in self.docs:
            vec = []
            for t in terms:
                if t in d:
                    vec.append(t.salience)
                else:
                    vec.append(0)
            vecs.append(vec)

        vecs = np.array(vecs)
        print(vecs.shape)
        print(vecs)
        return vecs
