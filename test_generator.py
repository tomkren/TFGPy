import unittest
from collections import OrderedDict

from generator_static import ts, subs, get_num
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
    def test_d(self):
        for goal, gamma in [d1(), d2()]:
            for k in range(1, 5):
                results = ts(gamma, k, goal, 0)
                num = get_num(gamma, k, goal)
                self.assertEqual(num, len(results))
                for t_res in results:
                    self.assertTrue(t_res.tree.is_well_typed(gamma))


if __name__ == "__main__":
    #unittest.main()

    def f():
        goal, gamma = d1()
        if False:
            print(gamma)
            print('=' * 20)
            print(goal)
            print('=' * 20)
        for k in range(1, 7):
            num = get_num(gamma, k, goal)
            results = ts(gamma, k, goal, 0)
            assert num == len(results)
            for t_res in results:
                assert t_res.tree.is_well_typed(gamma)

    f()
