import unittest
from collections import OrderedDict

from generator_static import ts
from parsers import parse_ctx, parse_typ


def make_gamma():
    return parse_ctx(OrderedDict([
        ("s", (("a", "->", ("b", "->", "c")), '->',
               (("a", "->", "b"), "->", ("a", "->", "c")))),
        ("k", ("a", "->", ("b", "->", "a"))),
        ("seri", (("Dag", 'a', 'b'), '->', (("Dag", 'b', 'c'), '->', ("Dag", 'a', 'c')))),
        ("para", (("Dag", 'a', 'b'), '->', (("Dag", 'c', 'd'), '->', ("Dag", ('P', 'a', 'c'), ('P', 'b', 'd'))))),
        ("mkDag", (("a", "->", "b"), '->', ("Dag", "a", "b"))),
        ("deDag", (("Dag", "a", "b"), '->', ("a", "->", "b"),)),
        ("mkP", ("a", "->", ("b", "->", ('P', "a", 'b')))),
        ("fst", (('P', "a", 'b'), '->', 'a')),
        ("snd", (('P', "a", 'b'), '->', 'b')),
    ]))


def make_goal():
    return parse_typ((('P', 'A', ('P', 'A', 'A')), '->', ('P', 'A', ('P', 'A', 'A'))))


class TestStaticGen(unittest.TestCase):
    def test(self):
        print(str(make_gamma()))
        print(str(make_goal()))
        print('='*20)

        gamma, goal = make_gamma(), make_goal()
        for k in range(1, 4):
            res = ts(gamma, k, goal, 0)
            for a in res:
                print(repr(a))
            pass


if __name__ == "__main__":
    unittest.main()

