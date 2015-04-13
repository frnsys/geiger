from scipy.spatial.distance import cosine, euclidean
from geiger.text import strip_tags
from geiger.models.doc2vec import Model as Doc2Vec

doc1 = '''
Supplements helped to save my life when I had a very deadly cancer 9 years ago and have helped keep me in remission since (I also had chemo and surgery, but my odds of survival at the time and continued longevity were very low.)  However, I have taken them under the guidance of a highly trained and very knowledgeable naturopathic doctor, who is scrupulous about using supplements whose manufacturers meet high standards.   I would hate to see  all supplements banned or restricted, but certainly the FDA should require all ingredients to be listed, all ingredients listed to actually be in the supplements (another issue that came up recently -- though fraud statutes alone should handle that), and any outright dangerous ingredients prohibited.
'''

doc2 = '''
I use natural supplements like vitamins and herbs. You just have to look at the packaging of these products to know there might be something risky going on. On the other hand, Ephedra, an herb used for thousands of years in Chinese medicine, became illegal because of  people using it wrongly, and having heart problems. For those of us who could use it sometimes for lung problems, it is not available. Still we could get these supplements...maybe regulation is good, if we don't lose the ability to get the supplements we use for being healthy.
'''

doc3 = '''
So will the AMA step up and pull the licenses of these corrupt physicians since the FDA is so polluted by conflicts of interests in this area that they are frozen?  Senator Hatchett Job should be investigated as a part of this overt fraud, along with his Hatchling.  Is every GOP Senator corrupted by something?  Really sick that our legislators are allowed to take our money for fair services and use them strictly to enrich themselves.
'''

from gensim.models.doc2vec import Doc2Vec

m = Doc2Vec()
print(dir(m))
print(type(m))

m = Doc2Vec.load_word2vec_format('/Users/ftseng/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
print(dir(m))
print(type(m))
vec1 = m.infer_vector(strip_tags(doc1))
vec2 = m.infer_vector(strip_tags(doc2))
vec3 = m.infer_vector(strip_tags(doc3))


#m = Doc2Vec()
#vec1 = m.infer_vector(strip_tags(doc1))
#vec2 = m.infer_vector(strip_tags(doc2))
#vec3 = m.infer_vector(strip_tags(doc3))

dist1 = euclidean(vec1, vec2)
dist2 = cosine(vec1, vec2)
print('Euclidean distance: {0}'.format(dist1))
print('Cosine distance: {0}'.format(dist2))

dist1 = euclidean(vec1, vec3)
dist2 = cosine(vec1, vec3)
print('Euclidean distance: {0}'.format(dist1))
print('Cosine distance: {0}'.format(dist2))

dist1 = euclidean(vec2, vec3)
dist2 = cosine(vec2, vec3)
print('Euclidean distance: {0}'.format(dist1))
print('Cosine distance: {0}'.format(dist2))
