import unittest
from collections import OrderedDict

from generator import Generator
from generator_static import ts, get_num
from parsers import parse_ctx, parse_typ

REALLY_SHORT_TIME = 0.01


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
            ])),
            4)


def d2():
    return (parse_typ('B'),
            parse_ctx(OrderedDict([
                ("f", ("A", "->", 'B')),
                ("x", "A"),
                ("y", "B"),
            ])),
            5)


class TestGen(unittest.TestCase):
    def test_d(self):
        for goal, gamma, max_k in [d1(), d2()]:
            g = Generator(gamma)
            res = []
            for k in range(1, max_k + 1):
                # check static generator
                s_num = get_num(gamma, k, goal)
                s_trees = ts(gamma, k, goal, 0)
                self.assertEqual(s_num, len(s_trees))
                for t in s_trees:
                    self.assertTrue(t.tree.is_well_typed(gamma))

                # check generator
                g_num = g.get_num(k, goal)
                self.assertEqual(s_num, g_num)
                res.append(g_num)

            # second run should have the same results
            # but it should be much faster
            start = time.time()
            for k in range(1, max_k + 1):
                g_num = g.get_num(k, goal)
                self.assertEqual(res[k - 1], g_num)
            end = time.time()
            self.assertLess(end - start, REALLY_SHORT_TIME)


if __name__ == "__main__":
    unittest.main()

    if False:
        import time

        goal, gamma = d1()
        g = Generator(gamma)
        if True:
            print(gamma)
            print('=' * 20)
            print(goal)
            print('=' * 20)
        for k in range(1, 7):
            num = g.get_num(k, goal)
            print(k, num)

        a = time.time()
        for k in range(1, 7):
            num = g.get_num(k, goal)
            print(k, num)
        print("%.2f" % (time.time() - a))
