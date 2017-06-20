from collections import OrderedDict
from functools import reduce
from itertools import product
from operator import xor

import app_tree
import domain_koza_apptree
from fitness_cache import FitnessCache
from parsers import parse_typ, parse_ctx

size_d = {}


def domain_parity(SIZE):
    values = product(*((True, False) for _ in range(SIZE)))
    ALL = [(bits, reduce(xor, bits)) for bits in values]

    global_symbols = {
        's_and': lambda x: (lambda y: x and y),
        's_or': lambda x: (lambda y: x or y),
        's_nand': lambda x: (lambda y: not (x and y)),
        's_nor': lambda x: (lambda y: not (x or y)),
        's_xor': lambda x: (lambda y: x ^ y)
    }

    R = 'R'
    goal = parse_typ(R)
    vars = ['b%d' % i for i in range(SIZE)]
    var_str = ','.join(vars)
    stuff = [('s_and', (R, '->', (R, '->', R))),
             ('s_or', (R, '->', (R, '->', R))),
             ('s_nor', (R, '->', (R, '->', R))),
             ('s_nand', (R, '->', (R, '->', R))),
             ('s_xor', (R, '->', (R, '->', R))),
             ] + [(v, R) for v in vars]

    gamma = parse_ctx(OrderedDict(stuff))
    cache = FitnessCache()

    def format_one(eval_str):
        return "lambda %s : %s" % (var_str, eval_str)

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
            if fun(*values) == result:
                score += 1

        cache.update(s, score)
        return score

    return goal, gamma, fitness, (lambda: len(cache)), cache


def make_env_app_tree(SIZE=6):
    return domain_koza_apptree.make_env_app_tree(get_raw_domain=lambda: domain_parity(SIZE),
                                                 early_end_limit=2 ** SIZE)
