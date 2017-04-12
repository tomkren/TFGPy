import unittest
from collections import OrderedDict

from generator_static import ts
from parsers import parse_ctx, parse_typ


def d1():
    return (parse_typ((('P', 'A', ('P', 'A', 'A')), '->', ('P', 'A', ('P', 'A', 'A')))),
            parse_ctx(OrderedDict([
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
            ])))


def d2():
    return (parse_typ('B'),
            parse_ctx(OrderedDict([
                ("f", ("A", "->", 'B')),
                ("x", "A"),
                ("y", "B"),
            ])))


class TestStaticGen(unittest.TestCase):
    def test(self):
        pass


if __name__ == "__main__":
    # unittest.main()

    goal, gamma = d1()
    print(gamma)
    print('=' * 20)
    print(goal)
    print('=' * 20)
    for k in range(1, 6):
        res = ts(gamma, k, goal, 0)
        print(k, ":", len(res))
        for a in res:
            print(a.tree)
            assert a.tree.is_well_typed(gamma)
        pass
