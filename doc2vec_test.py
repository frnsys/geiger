from geiger.models.doc2vec import Model as Doc2Vec

print('Loading Doc2Vec model...')
m = Doc2Vec()

print('Loaded model.')
print('Checking accuracy...')
results = m.m.accuracy('/Users/ftseng/Downloads/questions-words.txt')
print(results)
