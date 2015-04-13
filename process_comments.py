import pandas as pd
from geiger.text import strip_tags
from geiger.util.progress import Progress

print('Loading comments...')

bodies = []
for path in ['data/all_approved_comments_01.csv', 'data/all_approved_comments_02.csv', 'data/all_approved_comments_03.csv']:
    comments = pd.read_csv(path, index_col=0, lineterminator='\n')
    p = Progress()
    n = len(comments) - 1
    for i, row in enumerate(comments.iterrows()):
        p.print_progress(i/n)
        comment = row[1]
        bodies.append(strip_tags(comment.commentBody).replace('\n', ' ').replace('\r', ' '))

with open('data/commentBodies.txt', 'w') as f:
    f.write('\n'.join(bodies))

# sanity check
#with open('data/commentBodies.txt', 'r') as f:
    #assert len(bodies) == len(f.readlines())