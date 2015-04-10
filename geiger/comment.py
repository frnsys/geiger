from datetime import datetime
from geiger.text import strip_tags

class Comment():
    def __init__(self, comment_data):
        self.id = comment_data['commentID']
        self.body_html = comment_data['commentBody']
        self.body = strip_tags(self.body_html)
        self.score = comment_data['recommendations']
        self.author = comment_data['userDisplayName']
        self.replies = [Comment(r) for r in comment_data['replies']]
        self.created_at = datetime.fromtimestamp(int(comment_data['createDate']))
