import json
path_to_examples = 'data/examples.json'
clusters = json.load(open(path_to_examples, 'r'))

from geiger.aspects import summarize_aspects

class Doc():
    def __init__(self, doc):
        self.body = doc

docs = []
labels = []
for i, clus in enumerate(clusters):
    for doc in clus:
        docs.append(Doc(doc))
        labels.append(i)

summarize_aspects(docs)
