import unittest
import time
from collections import OrderedDict

import random
import sys

import normalization
import typ
from app_tree import UnfinishedLeaf
from cache import Cache, CacheNop
from domain_fparity_apptree import d_general_even_parity
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
                ("para", (("Dag", 'a', 'b'), '->', (("Dag", 'c', 'd'), '->', ("Dag", ('P', 'a', 'c'), ('P', 'b', 'd'))))),
                ("mkDag", (("a", "->", "b"), '->', ("Dag", "a", "b"))),
                ("deDag", (("Dag", "a", "b"), '->', ("a", "->", "b"),)),
                ("mkP", ("a", "->", ("b", "->", ('P', "a", 'b')))),
                ("fst", (('P', "a", 'b'), '->', 'a')),
                ("snd", (('P', "a", 'b'), '->', 'b')),
            ])),
            4)


def d_general_even_parity_sk():
    return (parse_typ('Bool'),
            parse_ctx(OrderedDict([

                ('xs', ('List', 'Bool')),

                ("s", (("a", "->", ("b", "->", "c")), '->', (("a", "->", "b"), "->", ("a", "->", "c")))),
                ("k", ("a", "->", ("b", "->", "a"))),

                ("and",  ('Bool', '->', ('Bool', '->', 'Bool'))),
                ("or",   ('Bool', '->', ('Bool', '->', 'Bool'))),
                ("nand", ('Bool', '->', ('Bool', '->', 'Bool'))),
                ("nor",  ('Bool', '->', ('Bool', '->', 'Bool'))),

                ('foldr', (('a', '->', ('b', '->', 'b')), '->', ('b', '->', (('List', 'a'), '->', 'b')))),

                ('true', 'Bool'),
                ('false', 'Bool')

                # ("head", (('List', 'Bool'), '->', ('Maybe', 'Bool'))),
                # ("tail", (('List', 'Bool'), '->', ('Maybe', ('List', 'Bool')))),
            ])),
            5)


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
    def test_d2(self):
        return
        for goal, gamma, max_k in [d_general_even_parity()]:#d1(), d2(), d3()]:
            g = Generator(gamma, normalizator=normalization.Normalizator)

            for k in range(1, max_k + 1):
                g_num = g.get_num(k, goal)
                print(g_num)

    def test_d(self):
        for goal, gamma, max_k in [d_general_even_parity(), d1(), d2(), d3()]:
            g = Generator(gamma, normalizator=normalization.NormalizatorNop)
            gnf = Generator(gamma, normalizator=normalization.Normalizator)
            gNC = Generator(gamma, normalizator=normalization.NormalizatorNop, cache=CacheNop)
            gnfNC = Generator(gamma, normalizator=normalization.Normalizator, cache=CacheNop)

            res = []
            for k in range(1, max_k + 1):
                # check static generator
                s_num = get_num(gamma, k, goal)
                s_trees = set(tr.tree for tr in ts(gamma, k, goal, 0))
                self.assertEqual(s_num, len(s_trees))
                for t in s_trees:
                    self.assertTrue(t.is_well_typed(gamma))

                # check generator
                g_num = g.get_num(k, goal)
                self.assertEqual(s_num, g_num)
                res.append(g_num)
                #print(g_num)

                # check generator in nf
                self.assertEqual(s_num, gnf.get_num(k, goal))
                for i in range(10):
                    t = gnf.gen_one(k, goal)
                    if s_num == 0:
                        self.assertIsNone(t)
                    else:
                        self.assertTrue(t.is_well_typed(gamma))

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

    def test_skeletons(self):
        check_skeletons(self)


IS_LOG_PRINTING = False


def set_log_printing(new_val=True):
    global IS_LOG_PRINTING
    IS_LOG_PRINTING = new_val


def log(*args):
    if IS_LOG_PRINTING:
        print(*args)
    else:
        pass


def check_skeletons(tester):
    for goal, gamma, max_k in [d1(), d2(), d3()]:
        log('goal:', goal)
        # gamma.add_internal_pair() # todo uplne smazat až bude fungovat
        g = Generator(gamma)
        for k in range(1, max_k+1):
            log(' k:', k)
            check_successors(tester, g, k, goal)


def check_successors(tester, generator, k, goal_typ):
    sk = UnfinishedLeaf()
    sk_smart = UnfinishedLeaf(goal_typ)
    all_trees = set(tr.tree for tr in ts(generator.gamma, k, goal_typ, 0))
    if all_trees:
        check_successors_acc(tester, generator, k, goal_typ, sk, sk_smart, all_trees)


def log_expansion(parent_skeleton, next_skeletons, start_time):
    delta_time = time.time() - start_time
    ss_str = ' ... ' + ', '.join((str(s) for s in next_skeletons)) if next_skeletons else ''
    num = str(len(next_skeletons))
    log('  dt=', '%.2f' % delta_time, parent_skeleton, (' --> num=' + num), ss_str)


def check_successors_acc(tester, generator, k, goal_typ, parent_skeleton, parent_skeleton_smart, all_trees):
    t = time.time()
    skeletons = parent_skeleton.successors(generator, k, goal_typ)
    log_expansion(parent_skeleton, skeletons, t)

    t = time.time()
    skeletons_smart = parent_skeleton_smart.successors_smart(generator, k)
    log_expansion(parent_skeleton_smart, skeletons_smart, t)

    tester.assertEqual(len(skeletons), len(skeletons_smart))
    tester.assertEqual([str(s) for s in skeletons], [str(s) for s in skeletons_smart])
    log()

    if len(skeletons_smart) > 0:

        tree_smart = generator.gen_one_uf_smart(parent_skeleton_smart, k)
        log('   eg:', str(tree_smart))

        tester.assertTrue(tree_smart.is_well_typed(generator.gamma))

    else:
        tester.assertEqual(len(all_trees), 1)
        return

    skeleton2trees = {}
    sk2sk_smart = {}

    for (sk, sk_smart) in zip(skeletons, skeletons_smart):
        # log('    ', sk)
        # log('    ', sk_smart)
        sk2sk_smart[sk] = sk_smart

    for tree in all_trees:
        has_skeleton = False
        for sk in skeletons:
            if sk.is_skeleton_of(tree):
                tester.assertFalse(has_skeleton)
                has_skeleton = True
                skeleton2trees.setdefault(sk, []).append(tree)
        tester.assertTrue(has_skeleton)

    if len(skeletons) != len(skeleton2trees):
        tester.assertEqual(len(skeletons), len(skeleton2trees))

    for sk, all_trees_new in skeleton2trees.items():
        check_successors_acc(tester, generator, k, goal_typ, sk, sk2sk_smart[sk], all_trees_new)


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


def separate_error_404():
    # seed = random.randint(0, sys.maxsize)
    seed = 7669612278400467845
    random.seed(seed)
    print(seed)

    goal, gamma, max_k = d3()
    gene = Generator(gamma)
    hax_k = 3
    hax_typ = parse_typ(('_P_', 4, (5, '->', (6, '->', 7))))
    hax_tree = gene.gen_one(hax_k, hax_typ)
    print(hax_tree.typ)


def separate_error_404_sub():

    goal, gamma, max_k = d3()
    gene = Generator(gamma)
    k = 1
    n = 4
    typ = parse_typ((1, '->', (2, '->', 3)))
    tree = gene.subs(k, typ, n)
    print(tree.typ)


def separate_error_ip_new():
    goal, gamma, max_k = d3()
    gene = Generator(gamma)
    k = 2
    skel = UnfinishedLeaf(goal)

    set_log_printing(True)

    t = time.time()
    next_skels = skel.successors_smart(gene, k)
    log_expansion(skel, next_skels, t)

    # print(next_skels)


def separate_error_bad_smart_expansion_2017_02_28():
    print('Separating error: bad_expansion_2017_02_28')
    problem_goal, problem_gamma, _ = d3()
    gene = Generator(problem_gamma)
    problem_k = 5
    skel_0 = UnfinishedLeaf(problem_goal)

    set_log_printing(True)

    def succ(sk, path=None, is_smart=True, goal_typ=None):
        t = time.time()
        if is_smart:
            next_sks = sk.successors_smart(gene, problem_k)
        else:
            next_sks = sk.successors(gene, problem_k, goal_typ)
        log_expansion(sk, next_sks, t)
        if not path:
            return next_sks
        else:
            i = path[0]
            path = path[1:]
            next_one = next_sks[i]
            print('  i=', i, 'selected:', next_one)
            return succ(next_one, path, is_smart, goal_typ) if path else next_one

    bug_path_1 = [0, 0, 0, 2, 0, 0]  # (((k (? ?)) ?) ?)
    bug_path_2 = [0, 0, 0, 2, 0, 0]

    skel = succ(skel_0, bug_path_1, False, problem_goal)
    print(skel)
    print()

    seed = 42
    random.seed(seed)
    print('seed:', seed)
    tree = gene.gen_one_uf(skel, problem_k, problem_goal)
    log(str(tree))
    log('is_well_typed:', tree.is_well_typed(gene.gamma))

    print()

    skel = succ(skel_0, bug_path_2)
    print(skel)
    print()


if __name__ == "__main__":
    if True:
        unittest.main()
        # separate_error_ip_new()
        # separate_error_404()
        # separate_error_404_sub()
        # separate_error_bad_smart_expansion_2017_02_28()

    else:

        # seed = random.randint(0, sys.maxsize)
        seed = 1482646273836000672
        # seed = 2659613674626116145
        # seed = 249273683574813401
        random.seed(seed)
        print(seed)
        # print('randomState:', random.getstate())

        IS_LOG_PRINTING = True
        check_skeletons(TestGen())

    if not True:
        goal, gamma, max_k = d2()

        # print(gamma, '\n')
        # gamma.add_internal_pair()  # todo uplne smazat až bude fungovat

        print(gamma, '\n')

        gen = Generator(gamma)
        k = 2

        skeleton = UnfinishedLeaf()
        skeleton_smart = UnfinishedLeaf(goal)

        succs = skeleton.successors(gen, k, goal)
        print('[', ','.join(str(s) for s in succs), ']')

        succs_smart = skeleton_smart.successors_smart(gen, k)
        print('[', ','.join(str(s) for s in succs_smart), ']')

        skeleton = succs[0]
        skeleton_smart = succs_smart[0]

        succs = skeleton.successors(gen, k, goal)
        print('[', ','.join(str(s) for s in succs), ']')

        succs_smart = skeleton_smart.successors_smart(gen, k)
        print('[', ','.join(str(s) for s in succs_smart), ']')

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

        goal, gamma, max_k = d3()  # d1()
        max_k = 2

        gen = Generator(gamma, cache=Cache, normalizator=Normalizator)

        if True:
            print(gamma)
            print('=' * 30)
            print(goal)
            print('=' * 30, '\n')


        def generate_stuff():
            a = time.time()
            for k in range(1, max_k + 1):

                print('-- k =', k, '-' * 30)

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
