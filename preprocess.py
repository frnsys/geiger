"""
Convert CSVs to simple text docs.
"""

import pandas as pd
from geiger.text import strip_tags
from geiger.util.progress import Progress

def main():
    print('Processing comments...')
    id = 0
    for path in ['data/nyt_comments/all_approved_comments_01.csv',
                'data/nyt_comments/all_approved_comments_02.csv',
                'data/nyt_comments/all_approved_comments_03.csv']:
        comments = pd.read_csv(path, index_col=0, lineterminator='\n')
        bodies = []
        p = Progress()
        n = len(comments) - 1
        for i, row in enumerate(comments.iterrows()):
            p.print_progress(i/n)
            comment = row[1]
            bodies.append(strip_tags(comment.commentBody).replace('\n', ' ').replace('\r', ' '))

            # Save every 1mil
            if len(bodies) % 1000000 == 0:
                with open('nyt_comments_{0}.txt'.format(id), 'w') as f:
                    f.write('\n'.join(bodies))
                    bodies = []
                    id += 1

        with open('nyt_comments_{0}.txt'.format(id), 'w') as f:
            f.write('\n'.join(bodies))
            bodies = []
            id += 1

    #print('Processing assets...')
    #assets = pd.read_csv('../nyt_assets/data/assets.csv', index_col=0, lineterminator='\n')
    #bodies = []
    #p = Progress()
    #n = len(assets) - 1
    #for i, row in enumerate(assets.iterrows()):
        #p.print_progress(i/n)
        #asset = row[1]
        #bodies.append(strip_tags(asset.assetBody).replace('\n', ' ').replace('\r', ' '))

    #with open('nyt_assets.txt', 'w') as f:
        #f.write('\n'.join(bodies))


if __name__ == '__main__':
    main()
