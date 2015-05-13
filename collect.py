import json
from geiger.services import get_comments


def get_replies(comments):
    for c in comments:
        yield {'body': c.body, 'score': c.score}
        if c.replies:
            yield from get_replies(c.replies)

comments = get_comments('http://www.nytimes.com/2015/04/23/us/opponents-of-gay-marriage-ponder-strategy-as-issue-reaches-supreme-court.html', n=1900)


data = [r for r in get_replies(comments)]
print(len(data))

with open('gaymarriage_example.json', 'w') as f:
    json.dump(data, f)
