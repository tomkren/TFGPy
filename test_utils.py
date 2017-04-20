import unittest
from collections import OrderedDict

import utils


def mk_table(a, b):
    assert len(a) == len(b)
    return OrderedDict(zip(a, b))


def tbij(tester, f, t, test_bf, test_bt):
    a, b = utils.construct_bijection(mk_table(f, t))

    tester.assertEqual(set(a.items()), set(zip(test_bf, test_bt)))
    tester.assertEqual(set(b.items()), set(zip(test_bt, test_bf)))


class TestBijection(unittest.TestCase):
    def test_bij(self):
        tbij(self,
             [0, 2, 4, 6],
             [3, 2, 1, 4],
             [0, 1, 3, 4, 6],
             [3, 6, 0, 1, 4])

    def test_bij2(self):
        tbij(self,
             [0, 1, 3, 4, 5],
             [0, 3, 4, 2, 1],
             [1, 2, 3, 4, 5],
             [3, 5, 4, 2, 1])


if __name__ == "__main__":
    unittest.main()
