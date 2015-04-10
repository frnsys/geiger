import json
path_to_examples = 'data/examples.json'
clusters = json.load(open(path_to_examples, 'r'))

from geiger.text import strip_tags
from geiger.aspects import summarize_aspects

class Doc():
    def __init__(self, doc):
        self.body = strip_tags(doc)

docs = []
labels = []
for i, clus in enumerate(clusters):
    for doc in clus:
        docs.append(Doc(doc))
        labels.append(i)

summarize_aspects(docs)
