import json
from geiger.models import lda
from geiger.text import strip_tags

def main():
    path_to_examples = 'data/examples.json'
    clusters = json.load(open(path_to_examples, 'r'))
    docs = []
    for clus in clusters:
        for doc in clus:
            docs.append(doc)


    docs = [strip_tags(doc) for doc in docs if len(doc) >= 140] # what is this, twitter?? drop short comments :D

    m = lda.Model(n_topics=5, verbose=True)
    clusters = m.cluster(docs)
    print(m.m.components_)
    return clusters

if __name__ == '__main__':
    main()
