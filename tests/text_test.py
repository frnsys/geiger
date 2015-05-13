import unittest
from geiger.text import clean, tokenize


class TextTests(unittest.TestCase):

    def test_clean(self):
        doc = 'HEY what\'s up check out this link?! http://foobar.com'
        expected = 'hey what up check out this link'
        cleaned = clean.clean_doc(doc)
        self.assertEqual(cleaned, expected)


    def test_strip_tags(self):
        doc = '<b>this is some <span class="yo">stuff</span></b>'
        expected = 'this is some  stuff'
        cleaned = clean.strip_tags(doc)
        self.assertEqual(cleaned, expected)
