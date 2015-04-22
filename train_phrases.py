from geiger.util.progress import Progress
from geiger.text import strip_punct
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Phrases

paths = ['data/asset_bodies.txt']

n = 0
for path in paths:
    print('Counting lines for {0}...'.format(path))
    n += sum(1 for line in open(path, 'r'))
print('Processing {0} lines...'.format(n))

def doc_stream(paths, n):
    """
    Generator to feed sentences to Doc2Vec.
    """

    i = 0
    p = Progress()
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                i += 1
                p.print_progress(i/n)
                for sent in sent_tokenize(line.lower()):
                    tokens = word_tokenize(strip_punct(sent))
                    yield tokens


docs = [
    ['the', 'new', 'york', 'times', 'is', 'a', 'newspaper'],
    ['concern', 'is', 'rising', 'in', 'many', 'quarters', 'that', 'the', 'united', 'states', 'is', 'retreating', 'from', 'global', 'economic', 'leadership', 'just', 'when', 'it', 'is', 'needed', 'most'],
    ['the', 'afghan', 'president', 'ashraf', 'ghani', 'blamed', 'the', 'islamic', 'state', 'group'],
    ['building', 'maintenance', 'by', 'the', 'hrynenko', 'family', 'which', 'owns', 'properties', 'in', 'the', 'east', 'village'],
    ['a', 'telegram', 'from', 'the', 'american', 'embassy', 'in', 'constantinople', 'to', 'the', 'state', 'department', 'in', 'washington']
]


# Change to use less memory. Default is 40m.
max_vocab_size = 40000000

# Train up to trigrams.
print('Training bigrams...')
bigram = Phrases(doc_stream(paths, n), max_vocab_size=max_vocab_size, threshold=8.)

print('Saving...')
bigram.save('bigram_model.phrases')

print('Training trigrams...')
trigram = Phrases(bigram[doc_stream(paths, n)], max_vocab_size=max_vocab_size, threshold=10.)

print('Saving...')
trigram.save('trigram_model.phrases')
print('Done.')


#print('Loading bigrams...')
#bigram = Phrases.load('bigram_model.phrases')

#print('Loading trigrams...')
#trigram = Phrases.load('trigram_model.phrases')

#for r in bigram[docs]:
    #print(r)

#for r in trigram[bigram[docs]]:
    #print(r)
