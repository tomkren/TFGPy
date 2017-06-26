from collections import OrderedDict
from functools import reduce
from itertools import product
from operator import xor

import app_tree
import domain_koza_apptree
from fitness_cache import FitnessCache
from parsers import parse_typ, parse_ctx

size_d = {}


def d_general_even_parity():
    return (parse_typ('Bool'),
            parse_ctx(OrderedDict([
                ("copy", ('a', '->', ('P', 'a', 'a'))),
                ("seri", (('a', '->', 'b'), '->', (('b', '->', 'c'), '->', ('a', '->', 'c')))),
                ("para", (('a', '->', 'b'), '->', (('c', '->', 'd'), '->', (('P', 'a', 'c'), '->', ('P', 'b', 'd'))))),

                ("s_and", (('P', 'Bool', 'Bool'), '->', 'Bool')),
                ("s_or", (('P', 'Bool', 'Bool'), '->', 'Bool')),
                ("s_nand", (('P', 'Bool', 'Bool'), '->', 'Bool')),
                ("s_nor", (('P', 'Bool', 'Bool'), '->', 'Bool')),

                ('foldr', ((('P', 'a', 'b'), '->', 'b'), '->', ('b', '->', (('List', 'a'), '->', 'b')))),
                ('True', 'Bool'),
                ('False', 'Bool'),
                ('xs', ('List', 'Bool')),
            ])),
            20)


def domain_parity(SIZE):
    values = product(*((True, False) for _ in range(SIZE)))
    ALL = [(bits, reduce(xor, bits)) for bits in values]

    global_symbols = {
        'copy': lambda x: (x, x),
        'seri': lambda f: (lambda g: (lambda a: g(f(a)))),
        'para': lambda f: (lambda g: (lambda t: (f(t[0], g(t[1]))))),
        's_and': lambda t: t[0] and t[1],
        's_or': lambda t: t[0] or t[1],
        's_nand': lambda t: not (t[0] and t[1]),
        's_nor': lambda t: not (t[0] or t[1]),
        'foldr': lambda f: (lambda acc: (lambda xs: reduce(lambda a, b: f((b, a)), reversed(xs), acc))),
    }

    goal, gamma, _ = d_general_even_parity()
    cache = FitnessCache()

    def format_one(eval_str):
        print(repr(eval_str))
        return "lambda xs : %s" % (eval_str)

    def fitness(individual_app_tree):
        global size_d
        size = individual_app_tree.count_nodes()[app_tree.Leaf]
        size_d[size] = size_d.get(size, 0) + 1

        s = format_one(individual_app_tree.eval_str())
        cres = cache.d.get(s, None)
        if cres is not None:
            return cres

        fun = eval(s, global_symbols)
        assert callable(fun)
        score = 0
        for values, result in ALL:
            if fun(values) == result:
                score += 1

        cache.update(s, score)
        return score

    return goal, gamma, fitness, (lambda: len(cache)), cache


def make_env_app_tree(SIZE=6):
    return domain_koza_apptree.make_env_app_tree(get_raw_domain=lambda: domain_parity(SIZE),
                                                 early_end_limit=2 ** SIZE)
