import re
import math
import numpy as np
from collections import Counter, defaultdict
from geiger.text.tokenize import extract_phrases, keyword_tokenize, gram_size, lemma_forms
from geiger.util.progress import Progress
from geiger.knowledge import W2V, IDF
from geiger.clusters import cluster

import config
w2v = W2V(remote=config.remote)
idf = IDF(remote=config.remote)


class SemSim():
    """
    Clusters tokenized documents by semantic similarity.

    A "term" is a keyword or a keyphrase.
    """

    def __init__(self, debug=False, min_salience=0.55):
        self.debug = debug
        self.min_salience = min_salience


    def _tokenize(self, raw_docs):
        """
        Return raw documents as lists of tokens.
        """
        # Remove keyphrases with more than 3 words to reduce runtime
        docs = [[t for t in keyword_tokenize(d) if gram_size(t) <= 3] for d in raw_docs]
        docs, keyphrases = extract_phrases(docs, raw_docs)

        return docs


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

        sims, sals = zip(*[(sim, (self.saliences[t1] + self.saliences[t2])/2) for t1, t2, sim, in pairs])

        return sum(sim * sal for sim, sal in zip(sims, sals))/sum(sals)


    def _term_sim(self, t1, t2):
        """
        Get word2vec similarity for two terms.
        """
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

        return (sal_c + sal_g)/2


    def _norm(self, x):
        n = (x-0.5)**2
        return math.exp(-n/0.05)


    def _semsim_pairs(self, d1, d2):
        """
        Construct maximally-semantically-similar pairs b/w terms of two documents

        Pairs are returned with the similarities so they don't need to be re-computed.
        """
        pairs = set()

        d1 = list(d1)
        d2 = list(d2)

        if not d1 or not d2:
            return pairs

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
            pairs.add((d1[i], d2[j], sub_mat[i,j]))

        # Max-sim pairs for d2->d1
        for j, i in enumerate(np.argmax(sub_mat, axis=0)):
            # Add in this order so we recognize duplicate pairs
            pairs.add((d1[i], d2[j], sub_mat[i,j]))

        return pairs


    def _prune(self, docs):
        """
        Aggressively prune noisy terms:
            - those that appear only in one document (IDF is 1.0)
            - those that are not sufficiently salient
            - those which are totally subsumed by a phrase
        This improves runtime and should improve output quality
        """
        redundant = {t for t in self.all_terms if gram_size(t) == 1}

        # This could be more efficient
        for doc in docs:
            cleared = set()
            for t in redundant:
                if t not in doc:
                    continue

                # If this term occurs outside of a phrase,
                # it is no longer a candidate
                n = doc.count(t)
                d = sum(1 for t_ in doc if t != t_ and t in t_)
                if n > d:
                    cleared.add(t)

            redundant = redundant.difference(cleared)

        if self.debug:
            print('Removed {0} redundant terms'.format(len(redundant)))

        pruned = []
        for doc in docs:
            pruned.append([t for t in doc if self.saliences[t] >= self.min_salience and self.iidf[t] < 1.0 and t not in redundant])
        return pruned


    def _preprocess(self, docs):
        self.raw_docs = docs

        if self.debug:
            print('clustering {0} docs'.format(len(self.raw_docs)))

        # Represent docs as list of terms
        self.docs = self._tokenize(self.raw_docs)

        # Compute intra-comment IDF and salience for all terms
        self.iidf = self._internal_idf(self.docs)
        self.all_terms = {t for terms in self.docs for t in terms}
        self.saliences = {t: self._salience(t) for t in self.all_terms}

        # testing
        self.all_terms_unfiltered = self.all_terms

        self.docs = self._prune(self.docs)
        self.all_terms = {t for terms in self.docs for t in terms}

        if self.debug:
            print('vocabulary has {0} terms'.format(len(self.all_terms)))


    def _distance_matrix(self):
        # Cache a w2v sim mat for faster lookup
        n = len(self.all_terms)
        self.w2v_sim_mat = np.full((n, n), -1)

        # Map terms to their indices in the w2v sim mat
        self.w2v_term_map = {t: i for i, t in enumerate(self.all_terms)}

        # Compute similarity matrix and convert to a distance matrix
        sim_mat = self._similarity_matrix(self.docs)
        sim_mat[np.where(sim_mat == 0)] = 0.000001
        dist_mat = 1/sim_mat - 1
        self.sim_mat = sim_mat

        return dist_mat


    # TO DO clean this up
    def cluster(self, docs, eps):
        self._preprocess(docs)
        dist_mat = self._distance_matrix()

        if self.debug:
            try:
                # Mean nearest distances
                mean_nd = np.mean(np.apply_along_axis(lambda a: np.min(a[np.nonzero(a)]), 1, dist_mat))
                print('mean nearest distance: {0}'.format(mean_nd))
            # If it so happens that all the distances are 1,
            # this will throw a ValueError
            except ValueError:
                pass


        # Represented docs in condensed form:
        # [(term, freq), ...]

        self.normalized_saliences = {}
        mxm = max(self.saliences.values())
        for k, v in self.saliences.items():
            self.normalized_saliences[k] = v/mxm

        condensed_docs = [[(t, f, self.normalized_saliences[t]) for t, f in list(Counter(d).items())] for d in self.docs]


        if self.debug:
            print('highlighting docs....')

        highlighted_docs = []
        for i, doc in enumerate(self.raw_docs):
            d = markup_highlights(doc, self.docs[i])
            highlighted_docs.append(d)

        clusters = cluster(dist_mat, eps, min_samples=3)
        if clusters:
            clusters = [[(
                i,
                self.raw_docs[i],
                highlighted_docs[i],
                sorted(condensed_docs[i], key=lambda t: self.saliences[t[0]], reverse=True)
            ) for i in clus] for clus in clusters]

        # Build descriptors for each cluster
        # TO DO clean this up
        descriptors = []
        for i, clus in enumerate(clusters):
            kw_sets = []
            for j, (idx, c, hi, kws) in enumerate(clus):
                kw_sets.append(set(kws))

            all_kw_counts = defaultdict(int)
            for kws in kw_sets:
                for kw in kws:
                    all_kw_counts[kw] += 1
            ranked_kws = [(kw, all_kw_counts[kw] * self.saliences[kw], nsal) for kw, freq, nsal in sorted(all_kw_counts.keys(), key=lambda k: all_kw_counts[k] * self.saliences[k[0]], reverse=True)]
            descriptors.append(ranked_kws)
        return clusters, descriptors



def markup_highlights(raw_doc, term_doc):
    """
    Highlights each instance of the given term
    in the document. All forms of the term will be highlighted.
    """
    doc = raw_doc
    term_doc = set(term_doc)
    term_doc = sorted(list(term_doc), key=lambda t: len(t), reverse=True) # Longest first
    for t in term_doc:
        for term in t.split(','):
            term = term.strip()

            # Determine which forms are present for the term in the document
            if gram_size(term) == 1:
                # Replace longer forms first so we don't replace their substrings.
                forms = sorted(lemma_forms(term, doc), key=lambda f: len(f), reverse=True)
            else:
                forms = [term]

            for t in forms:
                # This captures 'F.D.A' if given 'FDA'
                # yeah, it's kind of overkill
                reg_ = '[.]?'.join(list(t))

                # Spaces might be spaces, or they might be hyphens
                reg_ = reg_.replace(' ', '[\s-]')

                # Only match the term if it is not continguous with other characters.
                # Otherwise it might be a substring of another word, which we want to
                # ignore
                # The last matching group is to try and ignore things which are
                # in html tags.
                reg = '(^|{0})({1})($|{0})(?=[^>]*(<|$))'.format('[^A-Za-z]', reg_)

                if re.findall(reg, doc):
                    doc = re.sub(reg, '\g<1><span class="highlight" data-term="{0}">\g<2></span>\g<3>'.format(term), doc, flags=re.IGNORECASE)
                else:
                    # If none of the term was found, try with extra alpha characters
                    # This helps if a phrase was newly learned and only assembled in
                    # its lemma form, so we may be missing the actual form it appears in.
                    reg = '(^|{0})({1}[A-Za-z]?)()(?=[^>]*(<|$))'.format('[^A-Za-z]', reg_)
                    doc = re.sub(reg, '\g<1><span class="highlight" data-term="{0}">\g<2></span>\g<3>'.format(term), doc, flags=re.IGNORECASE)

    return doc
