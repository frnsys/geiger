from gensim.models.doc2vec import Doc2Vec

m = Doc2Vec.load_word2vec_format('/Users/ftseng/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

print(m.similarity('republican', 'libertarian'))
#Out[4]: -0.0055788494915927252

print(m.similarity('republican', 'democrat'))
#Out[5]: 0.1168960166507812

print(m.similarity('republican', 'cat'))
#Out[6]: -0.012395391797767143

print(m.similarity('iraq', 'iran'))
#Out[7]: -0.088751072443341322

print(m.similarity('iraq', 'china'))
#Out[8]: 0.016510800725083822

print(m.similarity('iran', 'china'))
#Out[9]: -0.023058148150531366
