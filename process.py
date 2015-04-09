import json
import os

clus = []
for fname in os.listdir('data/examples'):
    with open('data/examples/{0}'.format(fname), 'r') as f:
        if fname == 'misc.txt':
            clus += [[d.strip()] for d in f.read().split('\n') if d != '']
        else:
            clus.append([d.strip() for d in f.read().split('\n') if d != ''])

with open('data/examples.json', 'w') as f:
    json.dump(clus, f)