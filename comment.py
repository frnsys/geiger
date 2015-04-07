from datetime import datetime

class Comment():
    def __init__(self, comment_data):
        self.id = comment_data['commentID']
        self.body = comment_data['commentBody']
        self.score = comment_data['recommendations']
        self.author = comment_data['userDisplayName']
        self.replies = [Comment(r) for r in comment_data['replies']]
        self.created_at = datetime.fromtimestamp(int(comment_data['createDate']))
