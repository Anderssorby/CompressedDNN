from odin.compute.architecture import greedy
import numpy as np

import unittest


class TestArchitecture(unittest.TestCase):

    def test_greedy(self):
        scores = np.random.random_integers(0, 100, 100)

        def simple_constraint(j):
            j = list(map(int, j))
            return np.sum(scores[j])

        indexes = np.arange(0, 100, 1, dtype=np.int8)
        m_l = 50

        selected = greedy(constraint=simple_constraint, indexes=indexes, m_l=m_l, parallel=False)

        self.assertEqual(m_l, len(selected))

        selected = greedy(constraint=simple_constraint, indexes=indexes, m_l=m_l, parallel=True)
        self.assertEqual(m_l, len(selected))


