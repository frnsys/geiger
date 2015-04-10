import pandas as pd
from geiger.text import strip_tags
from geiger.util.progress import Progress

print('Loading comments...')
comments_path = 'data/comments.csv'
comments = pd.read_csv('data/comments.csv', index_col=0, lineterminator='\n')

# Load only approved comments.
comments = comments[comments.label == 1]
print('Loaded {0} comments.'.format(len(comments)))

with open('data/commentBodies.txt', 'w') as f:
    p = Progress()
    n = len(comments) - 1
    bodies = []
    for i, row in enumerate(comments.iterrows()):
        p.print_progress(i/n)
        comment = row[1]
        bodies.append(strip_tags(comment.commentBody).replace('\n', ' ').replace('\r', ' '))
    f.write('\n'.join(bodies))

# sanity check
#with open('data/commentBodies.txt', 'r') as f:
    #assert len(bodies) == len(f.readlines())