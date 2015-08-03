import json
from broca import Pipeline
from broca.tokenize.keyword import Overkill
from broca.preprocess import Cleaner, HTMLCleaner
from broca.vectorize import BoW
from geiger.pipes import LDA, SemSim, HSCluster, DBSCAN, AspectCluster, Distance

docs = [d['body'] for d in json.load(open('examples/climate_example.json', 'r'))]



preprocess = Pipeline(
    HTMLCleaner(),
    Cleaner()
)

pipelines = [
    Pipeline(
        preprocess,
        BoW(),
        LDA(n_topics=10),
        Distance(metric='euclidean'),
        [HSCluster(), DBSCAN()]
    ),
    Pipeline(
        preprocess,
        Overkill(),
        SemSim(),
        [HSCluster(), DBSCAN()]
    ),
    Pipeline(
        preprocess,
        BoW(),
        Distance(metric='euclidean'),
        [HSCluster(), DBSCAN()]
    ),
]

#for p in pipelines:
    #print('Running pipeline:', p)
    #outputs = p(docs)
    #doc_clusters = []
    #for out in outputs:
        #for clus in out:
            #clus_docs = []
            #for id in clus:
                #clus_docs.append(docs[id])
            #doc_clusters.append(clus_docs)
    #print(doc_clusters)
    #print('----------------------------------------')


from nltk.tokenize import sent_tokenize
from geiger.sentences import prefilter
from geiger.pipes.aspect import markup_highlights

# Get sentences, filtered fairly aggressively
sents = [[sent for sent in sent_tokenize(d) if prefilter(sent)] for d in docs]
sents = [sent for s in sents for sent in s]
aspect = Pipeline(
    preprocess,
    Overkill(),
    AspectCluster()
)

output = aspect(sents)
highlighted = []
for k, aspect_sents in output:
    highlighted.append([markup_highlights(k, sents[i]) for i in aspect_sents])
