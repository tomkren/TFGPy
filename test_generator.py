import unittest
import time
from collections import OrderedDict

import normalization
import typ
from cache import Cache, CacheNop
from normalization import Normalizator, NormalizatorNop
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
                ("para",
                 (("Dag", 'a', 'b'), '->', (("Dag", 'c', 'd'), '->', ("Dag", ('P', 'a', 'c'), ('P', 'b', 'd'))))),
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


def d3():
    return (parse_typ(('a', '->', 'b')),
            parse_ctx(OrderedDict([
                ("s", (("a", "->", ("b", "->", "c")), '->',
                       (("a", "->", "b"), "->", ("a", "->", "c")))),
                ("k", ("a", "->", ("b", "->", "a"))),
            ])),
            5)


class TestGen(unittest.TestCase):
    def test_d(self):
        for goal, gamma, max_k in [d1(), d2(), d3()]:
            g = Generator(gamma, normalizator=normalization.NormalizatorNop)
            gnf = Generator(gamma, normalizator=normalization.Normalizator)
            gNC = Generator(gamma, normalizator=normalization.NormalizatorNop, cache=CacheNop)
            gnfNC = Generator(gamma, normalizator=normalization.Normalizator, cache=CacheNop)

            res = []
            for k in range(1, max_k + 1):
                # check static generator
                s_num = get_num(gamma, k, goal)
                s_trees = set(ts(gamma, k, goal, 0))
                self.assertEqual(s_num, len(s_trees))
                for t in s_trees:
                    self.assertTrue(t.tree.is_well_typed(gamma))

                # check generator
                g_num = g.get_num(k, goal)
                self.assertEqual(s_num, g_num)
                res.append(g_num)

                # check generator in nf
                self.assertEqual(s_num, gnf.get_num(k, goal))
                for i in range(10):
                    t = gnf.gen_one(k, goal)
                    if s_num == 0:
                        self.assertIsNone(t)
                    else:
                        self.assertTrue(t.is_well_typed())
                        self.assertIn(t, s_trees)

                # check generator without cache
                self.assertEqual(s_num, gNC.get_num(k, goal))

                # check generator in nf without cache
                self.assertEqual(s_num, gnfNC.get_num(k, goal))

            # second run should have the same results
            # but it should be much faster
            start = time.time()
            for k in range(1, max_k + 1):
                g_num = g.get_num(k, goal)
                self.assertEqual(res[k - 1], g_num)
            end = time.time()
            self.assertLess(end - start, REALLY_SHORT_TIME)


def check_generators_have_same_outputs(generators, goal, max_k):

    def check_eq(xs):
        return all(x == xs[0] for x in xs)

    def check_eq_info(xs):
        if len(xs) == 0:
            return True

        head = xs[0]
        for x in xs:
            if x != head:
                print('!!!\n', str(x), '\n', str(head))
                return False
        return True

    for k in range(1, max_k + 1):
        print('-- k =', k, '-' * 30)
        sub_results_s = []
        for gen_name, gen in generators.items():
            print(' ', gen_name, '...', end='')
            sub_results = gen.subs(k, goal, 0)
            print('done')
            sub_results_s.append(sub_results)

        print(check_eq_info(sub_results_s))


if __name__ == "__main__":
    unittest.main()

    if not True:
        goal, gamma, max_k = d3()  # d1()
        # max_k = 2

        gens = {
            'gen_full': Generator(gamma, cache=Cache, normalizator=Normalizator),
            'gen_cache_only': Generator(gamma, cache=Cache, normalizator=NormalizatorNop),
            'gen_norm_only': Generator(gamma, cache=CacheNop, normalizator=Normalizator),
            'gen_lame': Generator(gamma, cache=CacheNop, normalizator=NormalizatorNop)
        }

        check_generators_have_same_outputs(gens, goal, max_k)

    if not True:
        import time

        goal, gamma, max_k = d3() # d1()
        max_k = 2

        gen = Generator(gamma, cache=Cache, normalizator=Normalizator)

        if True:
            print(gamma)
            print('=' * 30)
            print(goal)
            print('=' * 30, '\n')

        def generate_stuff():
            a = time.time()
            for k in range(1, max_k+1):

                print('-- k =', k, '-'*30)

                num = gen.get_num(k, goal)
                sub_results = gen.subs(k, goal, 0)

                print('NUM =', num, '\n')

                for sub_res in sub_results:
                    print(sub_res)

            print("\ntime: %.2f s\n" % (time.time() - a))

        generate_stuff()

        if False:
            print('=' * 40, '\n')
            generate_stuff()

