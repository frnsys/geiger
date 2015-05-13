import re
import string
from html.parser import HTMLParser


def clean_doc(doc):
    doc = doc.lower()

    # Remove URLs
    doc = re.sub(r'https?:\/\/.*[\r\n]*', '', doc, flags=re.MULTILINE)

    doc = doc.replace('\'s ', ' ')
    doc = strip_punct(doc)
    return doc.strip()


def strip_tags(html):
    # Any unwrapped text is ignored,
    # so wrap html tags just in case.
    # Looking for a more reliable way of stripping HTML...
    html = '<div>{0}</div>'.format(html)
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# Don't replace dashes, but do replace em-dashes with a space.
dash_map = {ord(p): ' ' for p in '—'}
punct_map = {ord(p): '' for p in (string.punctuation + '“”').replace('-', '')}
def strip_punct(doc):
    return doc.translate(dash_map).translate(punct_map)


def html_decode(s):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    from: <http://stackoverflow.com/a/275246/1097920>
    """
    for code in (
            ("'", '&#39;'),
            ('"', '&quot;'),
            ('>', '&gt;'),
            ('<', '&lt;'),
            ('&', '&amp;')
        ):
        s = s.replace(code[1], code[0])
    return s


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)
