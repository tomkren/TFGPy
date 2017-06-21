import copy
from functools import reduce
from itertools import product
from operator import xor

import domain_koza_stack
import utils
from domain_koza_apptree import Environment
from fitness_cache import FitnessCache
from stack import Stack
from tree_node import StackNode

size_d = {}


def domain_parity_stack(SIZE=6):
    values = product(*((True, False) for _ in range(SIZE)))
    ALL = [(bits, reduce(xor, bits)) for bits in values]

    def format_one(eval_str):
        return "lambda %s : %s" % (var_str, eval_str)

    def raiserun():
        raise RuntimeError()

    symbols_d = {
        's_xor': (2, lambda x, y: x ^ y),
        's_and': (2, lambda x, y: x and y),
        's_or': (2, lambda x, y: x or y),
        's_nand': (2, lambda x, y: not (x and y)),
        's_nor': (2, lambda x, y: not (x or y)),
    }
    vars = ['b%d' % i for i in range(SIZE)]
    var_str = ','.join(vars)
    symbols_d.update({v: (0, raiserun) for v in vars})
    sutils = Stack(symbols_d)
    cache = FitnessCache()

    def fitness(stack):
        global size_d
        size_d[len(stack)] = size_d.get(len(stack), 0) + 1

        assert sutils.count_missing(stack)[0] == 0
        s = repr(stack)

        cres = cache.d.get(s, None)
        if cres is not None:
            return cres

        def fun(*args):
            assert len(args) == len(vars)
            return sutils.eval_stack(stack, {var: val for var, val in zip(vars, args)})

        score = 0
        for values, result in ALL:
            if fun(*values) == result:
                score += 1

        cache.update(s, score)
        return score

    return sutils.finish, sutils.is_finished, sutils.successors, fitness, sutils.eval_stack, (lambda: len(cache)), cache


def make_env_stack(max_k=5, SIZE=6):
    return domain_koza_stack.make_env_stack(get_raw_domain=lambda: domain_parity_stack(SIZE),
                                            early_end_limit=2 ** SIZE)


if __name__ == "__main__":
    finish, is_finished, successors, fitness, eval_stack, count_evals, cache = domain_parity_stack()

    for i in range(200000):
        s = ['s_and', 's_xor']
        f = finish(s, 15)
        fitness(f)

    print('\n'.join("%s = %s" % (a, b) for a, b in sorted(size_d.items())))
