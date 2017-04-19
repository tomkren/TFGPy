import unittest
from collections import OrderedDict

import utils


def mk_table(a, b):
    assert len(a) == len(b)
    return OrderedDict(zip(a, b))


def tbij(tester, f, t, test_bf, test_bt):
    a, b = utils.construct_bijection(mk_table(f, t))

    tester.assertEqual(list(a.keys()), test_bf)
    tester.assertEqual(list(a.values()), test_bt)

    rev = sorted((v, k) for k, v in zip(test_bf, test_bt))
    tester.assertEqual(list(b.keys()), [t[0] for t in rev])
    tester.assertEqual(list(b.values()), [t[1] for t in rev])


class TestBijection(unittest.TestCase):
    def testBij(self):
        tbij(self,
                 [0, 2, 4, 6],
                 [3, 2, 1, 4],
                 [0, 1, 3, 4, 6],
                 [3, 6, 0, 1, 4])

    def testBij(self):
        tbij(self,
                 [0, 1, 3, 4, 5],
                 [0, 3, 4, 2, 1],
                 [1, 2, 3, 4, 5],
                 [3, 5, 4, 2, 1])


if __name__ == "__main__":
    unittest.main()
