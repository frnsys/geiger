import unittest
from geiger.semsim import SemSim


class SemSimTests(unittest.TestCase):
    def setUp(self):
        self.m = SemSim(debug=True)

    def test_similarity(self):
        docs = [
            '''
            Strange how the United States has to implement these costly environmental
            reforms while China, India and Russia somehow continue to pollute at will
            and grow their economies exponentially. Like the Iran nuclear negotiations
            Obama doesn't mind at all that America is getting taken to the cleaners while
            other nations get a pass.
            ''',
            '''
            Strange how the United States has to implement these costly environmental
            reforms while China, India and Russia somehow continue to pollute at will
            and grow their economies exponentially. Like the Iran nuclear negotiations
            Obama doesn't mind at all that America is getting taken to the cleaners while
            other nations get a pass.
            '''
        ]
        self.m.cluster(docs)
        self.assertEqual(self.m.sim_mat[0,1], 1.)
